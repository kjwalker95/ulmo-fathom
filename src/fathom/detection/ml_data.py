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
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from fathom.grams.lofar import LOFARGram, compute_lofar_gram
from fathom.models import LOFARConfig, StftConfig, SyntheticTruthManifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchExtractionConfig:
    """Patch grid + stride config. A2 baseline: 256×256 patches.

    Training stride 128 (50% overlap) for data augmentation across patches.
    Inference stride 64 (75% overlap) for line-stitching robustness.
    """
    patch_size: int = 256
    stride: int = 128


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
      - Compute (or fetch cached) LOFAR gram for the addressed clip
      - Slice patch at (f_start:f_start+ps, t_start:t_start+ps)
      - Derive labels from manifest.lines using mask_bin_indices + freq_curve_hz

    Inputs:
      clip_paths: list of WAV paths. Each must have a sibling
        <stem>.truth_manifest.json (SyntheticTruthManifest schema).
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
    ):
        if not clip_paths:
            raise ValueError("clip_paths is empty")
        self.lofar_config = lofar_config or default_lofar_config()
        self.patch_config = patch_config or PatchExtractionConfig()
        self._cache_size = cache_size
        self._gram_cache: dict[int, LOFARGram] = {}

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

        binary_label, heatmap_target = self._compute_labels(entry, gram, fs, ts)
        return patch_tensor, binary_label, heatmap_target

    def _compute_labels(
        self,
        entry: _ClipEntry,
        gram: LOFARGram,
        f_start: int,
        t_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive (binary_label, heatmap_target) from manifest lines."""
        ps = self.patch_config.patch_size
        heatmap = np.zeros(ps, dtype=np.float32)
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
                heatmap[gram_bin - f_start] = 1.0
                any_line = True

        binary_label = torch.tensor(1.0 if any_line else 0.0, dtype=torch.float32)
        return binary_label, torch.from_numpy(heatmap)

    def _get_gram(self, clip_idx: int, entry: _ClipEntry) -> LOFARGram:
        if clip_idx in self._gram_cache:
            return self._gram_cache[clip_idx]
        wav, _ = sf.read(str(entry.wav_path), always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        gram = compute_lofar_gram(wav.astype("float32"), self.lofar_config)
        if len(self._gram_cache) >= self._cache_size:
            self._gram_cache.pop(next(iter(self._gram_cache)))
        self._gram_cache[clip_idx] = gram
        return gram