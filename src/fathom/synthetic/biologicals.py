"""Biological confuser injection for synthetic LOFAR clips (A1 §3.2 + ENG feedback).

Loads a source-agnostic BiologicalClipLibrary, samples N biologicals per clip
via a categorical prior, and overlays each onto a running synthetic signal.
Per-overlay processing:
  - Load native-sample-rate WAV
  - Resample to target_sr (typically 32 kHz, matching synthetic ambient)
  - Apply a cosine taper at clip edges (suppresses ambient-mismatch artifacts:
    clip pads contain source-library ambient, synthetic base contains NOAA
    ambient; taper fades clip energy to zero at edges)
  - Scale amplitude to a sampled per-clip SNR against local ambient at the
    biological's freq_range center
  - Overlay at random t_onset within the synthetic clip's duration

Source-agnostic: future Watkins / other libraries load via the same
BiologicalClipLibrary manifest schema with no code change here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf

from fathom.ingestion._resample import resample_to
from fathom.models import BiologicalClip, BiologicalClipLibrary
from fathom.synthetic.tonals import (
    _cosine_taper_gate,
    _local_ambient_rms_at_frequency,
)

logger = logging.getLogger(__name__)


@dataclass
class BiologicalInjectionPriors:
    """Priors for biological overlay sampling."""

    # Number of biological overlays per clip (categorical).
    n_biologicals_distribution: Mapping[int, float] = field(
        default_factory=lambda: {0: 0.40, 1: 0.30, 2: 0.20, 3: 0.10}
    )
    # Per-overlay SNR (dB) vs local ambient at the bio's freq band; uniform.
    snr_db_range: tuple[float, float] = (3.0, 15.0)
    # Edge taper duration (s); cosine fade in + cosine fade out.
    taper_s: float = 0.3
    # Species sampling weights. If None, defaults to sqrt(library count) per
    # species — moderates extreme imbalances (e.g., DCLDE 2018's 1089 Bm vs 16 Eg).
    species_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        w_sum = sum(self.n_biologicals_distribution.values())
        if not np.isclose(w_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"n_biologicals_distribution weights must sum to 1.0; got {w_sum:.6f}"
            )
        if any(k < 0 for k in self.n_biologicals_distribution):
            raise ValueError("n_biologicals_distribution keys must be non-negative")


def load_biological_library(library_root: Path) -> BiologicalClipLibrary:
    """Read manifest.json from a library root directory."""
    manifest_path = library_root / "manifest.json"
    return BiologicalClipLibrary.model_validate_json(manifest_path.read_text())


def sample_n_biologicals(
    rng: np.random.Generator, priors: BiologicalInjectionPriors
) -> int:
    keys = sorted(priors.n_biologicals_distribution)
    weights = np.array([priors.n_biologicals_distribution[k] for k in keys], dtype=float)
    weights /= weights.sum()
    return int(rng.choice(keys, p=weights))


def _resolve_clip_audio(
    library_root: Path, clip: BiologicalClip, target_sr: int
) -> np.ndarray:
    """Load + mono-reduce + resample a clip to target_sr."""
    clip_path = library_root / clip.relative_path
    audio, source_sr = sf.read(str(clip_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if source_sr != target_sr:
        audio = resample_to(audio, source_sr, target_sr).astype(np.float32)
    return audio


def _sample_clip(
    rng: np.random.Generator,
    library: BiologicalClipLibrary,
    priors: BiologicalInjectionPriors,
) -> BiologicalClip:
    """Sample one clip: species (weighted) then uniform within species."""
    if priors.species_weights:
        species_list = sorted(
            set(library.species_counts) & set(priors.species_weights)
        )
        if not species_list:
            raise ValueError("priors.species_weights matches no species in library")
        weights = np.array(
            [priors.species_weights[s] for s in species_list], dtype=float
        )
    else:
        species_list = sorted(library.species_counts)
        counts = np.array(
            [library.species_counts[s] for s in species_list], dtype=float
        )
        weights = np.sqrt(counts)  # moderates Bm/Eg-style imbalance
    weights /= weights.sum()
    chosen_species = species_list[int(rng.choice(len(species_list), p=weights))]

    candidates = [c for c in library.clips if c.species_code == chosen_species]
    return candidates[int(rng.integers(0, len(candidates)))]


@dataclass(frozen=True)
class SampledBiologicalInjection:
    """Per-overlay metadata; consumed by the orchestrator to populate
    SyntheticConfuserLabel rows in the truth manifest."""

    clip_id: str
    species_code: str
    species_name: str
    source_dataset: str
    t_onset_s: float
    duration_s: float
    target_snr_db: float
    freq_range_hz: tuple[float, float]


def inject_biologicals(
    combined_signal: np.ndarray,
    sample_rate: int,
    *,
    library: BiologicalClipLibrary,
    library_root: Path,
    priors: BiologicalInjectionPriors,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[SampledBiologicalInjection]]:
    """Overlay 0-N biological confusers onto combined_signal.

    Each overlay is scaled to a sampled SNR vs local ambient at the species'
    freq_range center, tapered at the edges, and placed at a uniformly
    sampled t_onset within the clip duration. Returns the updated signal
    and per-overlay metadata.
    """
    if combined_signal.ndim != 1:
        raise ValueError(f"expected mono 1D signal; got shape {combined_signal.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive; got {sample_rate}")

    n = sample_n_biologicals(rng, priors)
    if n == 0 or len(library.clips) == 0:
        return combined_signal, []

    signal_duration_s = len(combined_signal) / sample_rate
    running = combined_signal.copy().astype(np.float32)
    overlays: list[SampledBiologicalInjection] = []

    for _ in range(n):
        clip = _sample_clip(rng, library, priors)
        bio_audio = _resolve_clip_audio(library_root, clip, sample_rate)
        bio_duration_s = len(bio_audio) / sample_rate

        if bio_duration_s >= signal_duration_s:
            logger.warning(
                "biological clip %s longer (%.2fs) than synthetic clip "
                "(%.2fs); skipping",
                clip.clip_id, bio_duration_s, signal_duration_s,
            )
            continue

        # Cosine taper to mask source-library-vs-synthetic ambient mismatch at edges
        taper_samples = max(1, int(priors.taper_s * sample_rate))
        taper_samples = min(taper_samples, max(1, len(bio_audio) // 2))
        gate = _cosine_taper_gate(len(bio_audio), 0, len(bio_audio), taper_samples)
        bio_tapered = (bio_audio * gate).astype(np.float32)

        bio_rms = float(np.sqrt(np.mean(bio_tapered ** 2)))
        if bio_rms <= 0:
            continue

        # SNR-anchored amplitude vs local ambient at the species' freq band center
        freq_center = 0.5 * (clip.freq_range_hz[0] + clip.freq_range_hz[1])
        local_rms = _local_ambient_rms_at_frequency(combined_signal, sample_rate, freq_center)
        if local_rms <= 0:
            logger.warning(
                "local ambient RMS at %.1f Hz is zero; skipping biological %s",
                freq_center, clip.clip_id,
            )
            continue

        target_snr_db = float(rng.uniform(*priors.snr_db_range))
        scale = (local_rms * (10.0 ** (target_snr_db / 20.0))) / bio_rms
        bio_scaled = (bio_tapered * scale).astype(np.float32)

        # Place at uniformly-sampled t_onset within the clip duration
        t_onset_s = float(rng.uniform(0.0, signal_duration_s - bio_duration_s))
        start_idx = int(t_onset_s * sample_rate)
        end_idx = start_idx + len(bio_scaled)
        running[start_idx:end_idx] = running[start_idx:end_idx] + bio_scaled

        overlays.append(SampledBiologicalInjection(
            clip_id=clip.clip_id,
            species_code=clip.species_code,
            species_name=clip.species_name,
            source_dataset=clip.source_dataset,
            t_onset_s=t_onset_s,
            duration_s=bio_duration_s,
            target_snr_db=target_snr_db,
            freq_range_hz=clip.freq_range_hz,
        ))

    return running, overlays