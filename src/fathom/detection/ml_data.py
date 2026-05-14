"""ML patch dataset for synthetic LOFAR clips (A2 §data-pipeline).

Consumes synthetic clip triplets (WAV + .truth_manifest.json) and produces
256×256 spectrogram patches with binary classification labels and
frequency-axis heatmap targets, computed on-the-fly per A2's data pipeline.

Patch grid:
  - Training stride: 128 (50% overlap)
  - Inference stride: 64 (75% overlap, per A2 §architecture)
  - Patch size: 256 × 256 (freq bins × time frames)

Label assignment from SyntheticTruthManifest:
  - binary_label = 1 if any line has an active (frame_idx, bin_idx) inside
    the patch window; else 0
  - heatmap_target (256-dim, freq axis): bin k = 1 if any line passes through
    LOFAR-gram freq bin (f_start + k) at any time within the patch window

Input representation: gram.normalized_power_db (split-window normalized,
same as the Sprint 2 classical detector consumes — apples-to-apples for the
C4 classical-vs-ML smoke and Sprint 5 agreement-confidence calibration).

Sprint 5 A2 (2026-05-13): pre-computed LOFAR gram support. If a sibling
`<stem>.lofar.npz` exists next to a WAV, _get_gram loads it instead of
running STFT. This eliminates the on-the-fly LOFAR computation that
bottlenecked A100 training. See scripts/precompute_lofar_grams.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

from fathom.grams.lofar import LOFARGram, compute_lofar_gram
from fathom.models import LOFARConfig, StftConfig, SyntheticTruthManifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PrecomputedGram:
    """Minimal LOFARGram stand-in for pre-computed `.lofar.npz` payloads.

    SyntheticPatchDataset only reads `normalized_power_db` and
    `frequencies_hz` from the gram. Other LOFARGram fields (times_s,
    power_db pre-norm, config) are not touched by the dataset, so the
    pre-compute saves only the two used arrays; this stub provides the
    duck-typed interface.
    """
    normalized_power_db: np.ndarray
    frequencies_hz: np.ndarray


@dataclass(frozen=True)
class PatchExtractionConfig:
    """Patch grid + stride config. A2 baseline: 256×256 patches.

    Training stride 128 (50% overlap) for data augmentation across patches.
    Inference stride 64 (75% overlap) for line-stitching robustness.

    target_mode controls the label shape:
      - "heatmap": (patch_size,) freq-axis 1D, for patch-CNN dual head
      - "mask":    (patch_size, patch_size) freq×time 2D, for U-Net segmentation
    """
    patch_size: int = 256
    stride: int = 128
    target_mode: str = "heatmap"


def default_lofar_config(sample_rate: int = 32000) -> LOFARConfig:
    """The canonical LOFARConfig used throughout Sprint 4 — matches what
    build_synthetic_c1_1.py and the C1.x demos used."""
    return LOFARConfig(
        stft=StftConfig(
            sample_rate=sample_rate,
            n_fft=16384,
            hop_length=4096,
            window_length=16384,
            window="hanning",
        ),
        freq_min_hz=3.0,
        freq_max_hz=1000.0,
        normalization_train_window_bins=33,
        normalization_central_window_bins=5,
        normalization_gap_bins=1,
    )


@dataclass(frozen=True)
class _ClipEntry:
    wav_path: Path
    manifest: SyntheticTruthManifest
    n_freq_bins: int
    n_time_frames: int


@dataclass(frozen=True)
class _PatchAddress:
    clip_idx: int
    f_start: int
    t_start: int


class SyntheticPatchDataset(Dataset):
    """LOFAR patches + (binary_label, heatmap_target) from synthetic triplets.

    Per __getitem__:
      - Compute (or fetch cached / pre-computed) LOFAR gram for the clip
      - Slice patch at (f_start:f_start+ps, t_start:t_start+ps)
      - Derive labels from manifest.lines using mask_bin_indices + freq_curve_hz

    Inputs:
      clip_paths: list of WAV paths. Each must have a sibling
        <stem>.truth_manifest.json (SyntheticTruthManifest schema). Optionally
        a <stem>.lofar.npz sibling skips on-the-fly STFT (Sprint 5 A2).
      lofar_config: LOFARConfig consumed by fathom.grams.lofar.compute_lofar_gram.
      patch_config: PatchExtractionConfig (default = training defaults).
      cache_size: number of grams to keep in memory (LRU).
    """

    def __init__(
        self,
        clip_paths: list[Path],
        lofar_config: LOFARConfig | None = None,
        patch_config: PatchExtractionConfig | None = None,
        cache_size: int = 16,
        transform=None,
    ):
        if not clip_paths:
            raise ValueError("clip_paths is empty")
        self.lofar_config = lofar_config or default_lofar_config()
        self.patch_config = patch_config or PatchExtractionConfig()
        self.transform = transform
        if self.patch_config.target_mode not in ("heatmap", "mask"):
            raise ValueError(
                f"target_mode must be 'heatmap' or 'mask'; "
                f"got {self.patch_config.target_mode!r}"
            )
        self._cache_size = cache_size
        self._gram_cache: dict[int, LOFARGram | _PrecomputedGram] = {}

        # Pre-derive gram shape per clip from STFT params + WAV duration; enumerate patches.
        stft = self.lofar_config.stft
        full_freqs = np.fft.rfftfreq(stft.n_fft, d=1.0 / stft.sample_rate)
        band_mask = (
            (full_freqs >= self.lofar_config.freq_min_hz)
            & (full_freqs <= self.lofar_config.freq_max_hz)
        )
        n_freq_bins = int(band_mask.sum())

        self._clip_entries: list[_ClipEntry] = []
        self._patch_addresses: list[_PatchAddress] = []

        ps = self.patch_config.patch_size
        stride = self.patch_config.stride

        for wav_path in clip_paths:
            wav_path = Path(wav_path)
            manifest_path = wav_path.with_name(wav_path.stem + ".truth_manifest.json")
            if not manifest_path.exists():
                logger.warning("missing manifest for %s; skipping", wav_path.name)
                continue
            manifest = SyntheticTruthManifest.model_validate_json(manifest_path.read_text())

            info = sf.info(str(wav_path))
            if info.frames < stft.window_length:
                logger.warning(
                    "WAV %s shorter than window_length; skipping", wav_path.name
                )
                continue
            n_time_frames = (info.frames - stft.window_length) // stft.hop_length + 1

            if n_freq_bins < ps or n_time_frames < ps:
                logger.warning(
                    "gram %dx%d smaller than patch_size %d; skipping %s",
                    n_freq_bins, n_time_frames, ps, wav_path.name,
                )
                continue

            clip_idx = len(self._clip_entries)
            self._clip_entries.append(_ClipEntry(
                wav_path=wav_path,
                manifest=manifest,
                n_freq_bins=n_freq_bins,
                n_time_frames=n_time_frames,
            ))

            f_starts = list(range(0, n_freq_bins - ps + 1, stride))
            if (n_freq_bins - ps) % stride != 0:
                f_starts.append(n_freq_bins - ps)
            t_starts = list(range(0, n_time_frames - ps + 1, stride))
            if (n_time_frames - ps) % stride != 0:
                t_starts.append(n_time_frames - ps)
            for fs in f_starts:
                for ts in t_starts:
                    self._patch_addresses.append(_PatchAddress(clip_idx, fs, ts))

        if not self._patch_addresses:
            raise ValueError(
                "no patches enumerated; check clip sizes vs patch_size"
            )

    def __len__(self) -> int:
        return len(self._patch_addresses)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        addr = self._patch_addresses[idx]
        entry = self._clip_entries[addr.clip_idx]
        gram = self._get_gram(addr.clip_idx, entry)

        ps = self.patch_config.patch_size
        fs, ts = addr.f_start, addr.t_start
        patch = gram.normalized_power_db[fs:fs + ps, ts:ts + ps].astype(np.float32)
        # (freq, time) → (1, freq, time) for PyTorch (C, H, W) convention
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)

        binary_label, target = self._compute_labels(entry, gram, fs, ts)
        if self.transform is not None:
            patch_tensor, target = self.transform(patch_tensor, target)
        return patch_tensor, binary_label, target

    def _compute_labels(
        self,
        entry: _ClipEntry,
        gram: LOFARGram | _PrecomputedGram,
        f_start: int,
        t_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive (binary_label, target) from manifest lines.

        target shape depends on patch_config.target_mode:
          - "heatmap": (patch_size,) 1D freq-axis target (max-projected over time)
          - "mask":    (patch_size, patch_size) freq×time 2D segmentation target
        """
        ps = self.patch_config.patch_size
        mode = self.patch_config.target_mode

        # Build the 2D mask first; "heatmap" mode reduces along the time axis at the end.
        mask = np.zeros((ps, ps), dtype=np.float32)
        any_line = False

        gram_freqs = gram.frequencies_hz  # LOFAR-masked freq axis
        for line in entry.manifest.lines:
            if not line.mask_bin_indices or not line.freq_curve_hz:
                continue
            for k, (frame_idx, _full_stft_bin) in enumerate(line.mask_bin_indices):
                if not (t_start <= frame_idx < t_start + ps):
                    continue
                # Re-derive LOFAR-masked gram bin from freq_curve_hz; robust to
                # any band-mask offset (mask_bin_indices' bin is in full-STFT space).
                freq_hz = float(line.freq_curve_hz[k])
                gram_bin = int(np.argmin(np.abs(gram_freqs - freq_hz)))
                if not (f_start <= gram_bin < f_start + ps):
                    continue
                mask[gram_bin - f_start, frame_idx - t_start] = 1.0
                any_line = True

        binary_label = torch.tensor(1.0 if any_line else 0.0, dtype=torch.float32)
        if mode == "mask":
            return binary_label, torch.from_numpy(mask)
        # "heatmap": time-axis OR-projection -> (patch_size,)
        heatmap = mask.max(axis=1)
        return binary_label, torch.from_numpy(heatmap)

    def _get_gram(
        self, clip_idx: int, entry: _ClipEntry
    ) -> LOFARGram | _PrecomputedGram:
        if clip_idx in self._gram_cache:
            return self._gram_cache[clip_idx]

        npz_path = entry.wav_path.with_suffix(".lofar.npz")
        if npz_path.exists():
            data = np.load(npz_path)
            gram: LOFARGram | _PrecomputedGram = _PrecomputedGram(
                normalized_power_db=data["normalized_power_db"],
                frequencies_hz=data["frequencies_hz"],
            )
        else:
            wav, _ = sf.read(str(entry.wav_path), always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            gram = compute_lofar_gram(wav.astype("float32"), self.lofar_config)

        if len(self._gram_cache) >= self._cache_size:
            self._gram_cache.pop(next(iter(self._gram_cache)))
        self._gram_cache[clip_idx] = gram
        return gram

    def get_all_binary_labels(self) -> list[bool]:
        """Precompute binary labels for all patches without loading audio.

        Needed by `make_balanced_patch_sampler` (which needs labels to weight
        samples). Uses precomputed freq axis instead of full LOFAR gram —
        ~1000× faster than calling __getitem__ for each patch.
        """
        stft = self.lofar_config.stft
        full_freqs = np.fft.rfftfreq(stft.n_fft, d=1.0 / stft.sample_rate)
        band_mask = (
            (full_freqs >= self.lofar_config.freq_min_hz)
            & (full_freqs <= self.lofar_config.freq_max_hz)
        )
        gram_freqs = full_freqs[band_mask]

        ps = self.patch_config.patch_size
        labels: list[bool] = []
        for addr in self._patch_addresses:
            entry = self._clip_entries[addr.clip_idx]
            f_start, t_start = addr.f_start, addr.t_start
            any_line = False
            for line in entry.manifest.lines:
                if any_line:
                    break
                if not line.mask_bin_indices or not line.freq_curve_hz:
                    continue
                for k, (frame_idx, _) in enumerate(line.mask_bin_indices):
                    if not (t_start <= frame_idx < t_start + ps):
                        continue
                    gram_bin = int(np.argmin(np.abs(gram_freqs - float(line.freq_curve_hz[k]))))
                    if f_start <= gram_bin < f_start + ps:
                        any_line = True
                        break
            labels.append(any_line)
        return labels


def make_balanced_patch_sampler(
    binary_labels: list[bool],
    num_samples: int,
) -> WeightedRandomSampler:
    """WeightedRandomSampler that draws ~50/50 positive/negative patches.

    Per-sample weight: 1/n_positives for positives, 1/n_negatives for negatives.
    With replacement, so `num_samples` can exceed dataset size — A2 specifies
    20k patches/epoch, which exceeds our ~7k unique patches and re-samples
    each ~3× per epoch (standard practice for small datasets).
    """
    n_total = len(binary_labels)
    if n_total == 0:
        raise ValueError("binary_labels is empty")
    n_pos = sum(binary_labels)
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"need both positive and negative samples; got pos={n_pos}, neg={n_neg}"
        )
    weight_pos = 1.0 / n_pos
    weight_neg = 1.0 / n_neg
    weights = [weight_pos if label else weight_neg for label in binary_labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
    )
