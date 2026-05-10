"""Deterministic tonal injection.

B1 minimum viable: single tonal at known params (no harmonics, no drift, no
clusters). C1 expands to the full A1 §3.3 parameterized schema with
decaying-cosine pulses + harmonic structure + drift.
"""
from __future__ import annotations

import numpy as np


def inject_deterministic_tonal(
    ambient: np.ndarray,
    sample_rate: int,
    *,
    frequency_hz: float,
    t_start_s: float,
    t_end_s: float,
    target_snr_db: float,
) -> tuple[np.ndarray, dict]:
    """Inject a pure-sinusoid tonal into ambient at known parameters.

    SNR is computed against global ambient RMS as a B1 simplification — the
    full per-frequency-bin local-ambient SNR computation (matching the
    classical detector convention per PCD v3 §6.3) lands in C1.

    Returns (combined_waveform, ground_truth_dict).
    """
    if ambient.ndim != 1:
        raise ValueError(f"expected mono 1D ambient; got shape {ambient.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive; got {sample_rate}")

    n = len(ambient)
    duration_s = n / sample_rate
    if t_start_s < 0 or t_end_s > duration_s or t_start_s >= t_end_s:
        raise ValueError(
            f"tonal time window [{t_start_s}, {t_end_s}] invalid for "
            f"ambient duration {duration_s:.2f}s"
        )

    ambient_rms = float(np.sqrt(np.mean(ambient ** 2)))
    if ambient_rms <= 0:
        raise ValueError("ambient is silent; cannot compute SNR-relative tonal amplitude")
    target_amplitude = ambient_rms * (10 ** (target_snr_db / 20))

    t = np.arange(n) / sample_rate
    tonal = (target_amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype("float32")
    gate = ((t >= t_start_s) & (t < t_end_s)).astype("float32")
    tonal *= gate

    combined = (ambient + tonal).astype("float32")

    ground_truth = {
        "frequency_hz": float(frequency_hz),
        "t_start_s": float(t_start_s),
        "t_end_s": float(t_end_s),
        "persistence_s": float(t_end_s - t_start_s),
        "target_snr_db": float(target_snr_db),
        "computed_amplitude": float(target_amplitude),
        "ambient_rms": ambient_rms,
    }
    return combined, ground_truth