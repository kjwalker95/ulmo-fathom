"""Parameter priors and sampler for C1.1 synthetic tonal injection (A1 §3.3 + deltas).

A1 §3.3 specifies parameter distributions for ship-tonal synthesis. C1.1 carries
five documented deltas vs A1 as written; see plan file
`~/.claude/plans/i-m-continuing-the-ulmo-synthetic-iverson.md` "A1 §3.3 deltas to log".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TonalParameterPriors:
    """A1 §3.3 priors with C1.1 deltas as default-valued fields."""

    # Fundamental frequency: bimodal uniform with primary-band bias.
    # delta vs A1: primary band low edge lifted from 5 Hz to 3 Hz (Phase 1 baseline freq_min).
    f0_primary_hz: tuple[float, float] = (3.0, 500.0)
    f0_secondary_hz: tuple[float, float] = (500.0, 1000.0)
    f0_primary_weight: float = 0.7

    # Harmonic structure.
    n_harmonics_choices: tuple[int, ...] = (1, 2, 3)
    harmonic_decay_range: tuple[float, float] = (0.3, 0.7)

    # Per-pulse decay (s^-1), sampled log-uniformly.
    decay_constant_log_range: tuple[float, float] = (0.01, 1.0)

    # Cluster timing (s), sampled log-uniformly. Within-cluster jitter is
    # Rayleigh(sigma=0.1*T); inter-cluster gap is Gaussian(0, 0.05*T).
    cluster_period_log_range: tuple[float, float] = (1.0, 60.0)

    # Total source persistence (s), sampled log-uniformly.
    total_persistence_log_range: tuple[float, float] = (1.0, 120.0)

    # Frequency drift rate (Hz/s), Gaussian.
    drift_rate_std_hz_per_s: float = 0.05

    # Per-source SNR (dB) — Normal in dB-units (A1's "log-normal" reading,
    # since dB is already a logarithmic scale).
    snr_log_mean_db: float = 8.0
    snr_log_std_db: float = 4.0

    # Pulses per cluster, inclusive uniform integer range.
    # delta vs A1: A1 silent on pulses-per-cluster. Operational interpretation.
    pulses_per_cluster_range: tuple[int, int] = (1, 5)

    # Source-count distribution per clip. Keys must be non-negative ints; values must sum to 1.
    # delta vs A1: A1 silent. Weighted to match real ship density and include negatives
    # required by C2 binary classifier.
    n_sources_distribution: Mapping[int, float] = field(
        default_factory=lambda: {0: 0.15, 1: 0.40, 2: 0.30, 3: 0.15}
    )

    # Multi-source f0 separation (Hz). Rejection threshold so two sources never
    # share a near-identical fundamental.
    # delta vs A1: A1 silent on multi-source.
    min_freq_separation_hz: float = 20.0
    freq_separation_max_retries: int = 32

    def __post_init__(self) -> None:
        # Validate n_sources_distribution sums to 1.0 within tolerance.
        weight_sum = sum(self.n_sources_distribution.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"n_sources_distribution weights must sum to 1.0; got {weight_sum:.6f}"
            )
        if any(k < 0 for k in self.n_sources_distribution):
            raise ValueError("n_sources_distribution keys must be non-negative integers")
        if any(v < 0 for v in self.n_sources_distribution.values()):
            raise ValueError("n_sources_distribution weights must be non-negative")
        if not (0.0 <= self.f0_primary_weight <= 1.0):
            raise ValueError("f0_primary_weight must be in [0, 1]")



@dataclass
class PropagationGeometryPriors:
    """C1.3-lite propagation geometry priors.

    A1 §3.4 specified pre-computed KRAKEN/BELLHOP IR libraries (5 envs ×
    10 geometries × 2 bands = 100 IRs). C1.3-lite substitutes a parametric
    three-path channel (direct + surface bounce + bottom bounce) with
    geometry sampled from these priors. See `src/fathom/synthetic/propagation.py`
    for the channel model and `generator.py:C1_3_LITE_DELTAS` for the four
    documented A1 §3.4 deltas.
    """

    # Water depth (m): uniform over a band that spans continental shelf
    # (~100 m) through abyssal basin (~4000 m).
    water_depth_m_range: tuple[float, float] = (100.0, 4000.0)

    # Source depth (m): surface vessel (5 m) through deep-running submarine
    # (200 m). delta vs A1: A1 §3.4 silent on per-clip depth sampling.
    source_depth_m_range: tuple[float, float] = (5.0, 200.0)

    # Receiver depth (m): towed array shallow (50 m) through deep mooring
    # (500 m). Bounded by water_depth at sample time.
    receiver_depth_m_range: tuple[float, float] = (50.0, 500.0)

    # Horizontal range (m): IUSS operational regime — first convergence
    # zone inner edge (~5 km) to outer detection edge (~50 km).
    horizontal_range_m_range: tuple[float, float] = (5_000.0, 50_000.0)

    # Sound speed (m/s): isovelocity baseline. delta vs A1: A1 §3.4
    # (KRAKEN) would model SSP-dependent refraction; deferred to Sprint 5+.
    sound_speed_m_per_s: float = 1500.0

    # Bottom reflection loss (dB per bounce): 3 dB (sandy/soft sediment)
    # to 10 dB (silty/lossy). Hard-rock reflectors (<1 dB) omitted as
    # edge case for lite scope.
    bottom_reflection_loss_db_range: tuple[float, float] = (3.0, 10.0)

    # Rejection-sample retries to enforce source_depth ≤ water_depth and
    # receiver_depth ≤ water_depth simultaneously.
    depth_rejection_max_retries: int = 32

    def __post_init__(self) -> None:
        for name, rng_pair in (
            ("water_depth_m_range", self.water_depth_m_range),
            ("source_depth_m_range", self.source_depth_m_range),
            ("receiver_depth_m_range", self.receiver_depth_m_range),
            ("horizontal_range_m_range", self.horizontal_range_m_range),
            ("bottom_reflection_loss_db_range", self.bottom_reflection_loss_db_range),
        ):
            lo, hi = rng_pair
            if not (lo > 0 and hi > 0):
                raise ValueError(f"{name} must be strictly positive; got {rng_pair}")
            if lo > hi:
                raise ValueError(f"{name} low > high: {rng_pair}")
        if self.sound_speed_m_per_s <= 0:
            raise ValueError("sound_speed_m_per_s must be > 0")
        if self.depth_rejection_max_retries < 1:
            raise ValueError("depth_rejection_max_retries must be >= 1")

@dataclass(frozen=True)
class SampledTonalParameters:
    """Concrete parameter set for one synthetic source (one fundamental + its harmonics)."""

    f0_hz: float
    n_harmonics: int
    harmonic_decay: float
    decay_constant_per_s: float
    cluster_period_s: float
    total_persistence_s: float
    drift_rate_hz_per_s: float
    target_snr_db: float
    t_onset_s: float


def sample_n_sources(rng: np.random.Generator, priors: TonalParameterPriors) -> int:
    """Categorical draw from priors.n_sources_distribution."""
    keys = sorted(priors.n_sources_distribution)
    weights = np.array([priors.n_sources_distribution[k] for k in keys], dtype=float)
    weights /= weights.sum()  # defensive renormalize
    return int(rng.choice(keys, p=weights))


def _sample_f0_hz(rng: np.random.Generator, priors: TonalParameterPriors) -> float:
    """Bimodal-uniform sample with primary-band bias."""
    if rng.uniform() < priors.f0_primary_weight:
        lo, hi = priors.f0_primary_hz
    else:
        lo, hi = priors.f0_secondary_hz
    return float(rng.uniform(lo, hi))


def sample_tonal_parameters(
    rng: np.random.Generator,
    priors: TonalParameterPriors,
    clip_duration_s: float,
    *,
    prior_f0s_hz: tuple[float, ...] = (),
) -> SampledTonalParameters | None:
    """Draw one SampledTonalParameters under priors. Returns None if f0
    rejection sampling exhausts retries given prior_f0s_hz.

    Uses log-uniform priors for decay, cluster period, and persistence;
    Gaussian for drift; Normal-in-dB for SNR. f0 is bimodal-uniform with
    rejection sampling against prior_f0s_hz at min_freq_separation_hz.
    """

    # --- f0 with rejection sampling against already-drawn sources ---
    f0_hz: float | None = None
    for _ in range(priors.freq_separation_max_retries + 1):
        candidate = _sample_f0_hz(rng, priors)
        if all(abs(candidate - prev) >= priors.min_freq_separation_hz for prev in prior_f0s_hz):
            f0_hz = candidate
            break
    if f0_hz is None:
        logger.warning(
            "f0 rejection sampling exhausted %d retries against prior_f0s=%s; "
            "skipping this source",
            priors.freq_separation_max_retries,
            list(prior_f0s_hz),
        )
        return None

    # --- harmonics ---
    n_harmonics = int(rng.choice(priors.n_harmonics_choices))
    harmonic_decay = float(rng.uniform(*priors.harmonic_decay_range))

    # --- log-uniform draws ---
    decay_lo, decay_hi = priors.decay_constant_log_range
    decay_constant_per_s = float(10.0 ** rng.uniform(np.log10(decay_lo), np.log10(decay_hi)))

    cp_lo, cp_hi = priors.cluster_period_log_range
    cluster_period_s = float(10.0 ** rng.uniform(np.log10(cp_lo), np.log10(cp_hi)))

    pers_lo, pers_hi = priors.total_persistence_log_range
    total_persistence_s = float(
        10.0 ** rng.uniform(np.log10(pers_lo), np.log10(pers_hi))
    )

    # Clamp persistence to clip duration; place onset uniformly in remaining window.
    if total_persistence_s > clip_duration_s:
        total_persistence_s = clip_duration_s
    onset_max = max(0.0, clip_duration_s - total_persistence_s)
    t_onset_s = float(rng.uniform(0.0, onset_max)) if onset_max > 0 else 0.0

    # --- drift + SNR ---
    drift_rate_hz_per_s = float(rng.normal(0.0, priors.drift_rate_std_hz_per_s))
    target_snr_db = float(rng.normal(priors.snr_log_mean_db, priors.snr_log_std_db))

    return SampledTonalParameters(
        f0_hz=f0_hz,
        n_harmonics=n_harmonics,
        harmonic_decay=harmonic_decay,
        decay_constant_per_s=decay_constant_per_s,
        cluster_period_s=cluster_period_s,
        total_persistence_s=total_persistence_s,
        drift_rate_hz_per_s=drift_rate_hz_per_s,
        target_snr_db=target_snr_db,
        t_onset_s=t_onset_s,
    )


@dataclass(frozen=True)
class SampledPropagationGeometry:
    """Concrete geometry for one C1.3-lite three-path channel realization."""

    water_depth_m: float
    source_depth_m: float
    receiver_depth_m: float
    horizontal_range_m: float
    sound_speed_m_per_s: float
    bottom_reflection_loss_db: float

def sample_propagation_geometry(
    rng: np.random.Generator,
    priors: PropagationGeometryPriors,
) -> SampledPropagationGeometry | None:
    """Draw one SampledPropagationGeometry. Rejection-samples until both
    source_depth and receiver_depth fall at or below the sampled water_depth.
    Returns None if retries exhaust (e.g., priors range edges are inconsistent)."""

    water_lo, water_hi = priors.water_depth_m_range
    src_lo, src_hi = priors.source_depth_m_range
    rcv_lo, rcv_hi = priors.receiver_depth_m_range

    for _ in range(priors.depth_rejection_max_retries + 1):
        water_depth_m = float(rng.uniform(water_lo, water_hi))
        # Cap source and receiver depth priors to water_depth before draw.
        eff_src_hi = min(src_hi, water_depth_m)
        eff_rcv_hi = min(rcv_hi, water_depth_m)
        if eff_src_hi < src_lo or eff_rcv_hi < rcv_lo:
            continue  # water_depth too shallow for the chosen receiver/source bands
        source_depth_m = float(rng.uniform(src_lo, eff_src_hi))
        receiver_depth_m = float(rng.uniform(rcv_lo, eff_rcv_hi))
        break
    else:
        logger.warning(
            "propagation geometry depth rejection exhausted %d retries (water=%s, "
            "source=%s, receiver=%s)",
            priors.depth_rejection_max_retries,
            priors.water_depth_m_range,
            priors.source_depth_m_range,
            priors.receiver_depth_m_range,
        )
        return None

    rng_lo, rng_hi = priors.horizontal_range_m_range
    horizontal_range_m = float(rng.uniform(rng_lo, rng_hi))

    btm_lo, btm_hi = priors.bottom_reflection_loss_db_range
    bottom_reflection_loss_db = float(rng.uniform(btm_lo, btm_hi))

    return SampledPropagationGeometry(
        water_depth_m=water_depth_m,
        source_depth_m=source_depth_m,
        receiver_depth_m=receiver_depth_m,
        horizontal_range_m=horizontal_range_m,
        sound_speed_m_per_s=priors.sound_speed_m_per_s,
        bottom_reflection_loss_db=bottom_reflection_loss_db,
    )