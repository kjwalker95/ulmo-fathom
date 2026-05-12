"""Parametric three-path underwater channel for C1.3-lite (A1 §3.4 substitute).

A1 §3.4 specified a pre-computed KRAKEN/BELLHOP IR library. The team's
canonical-IR hunt (2026-05-12) confirmed no public IR dataset covers the
3-1000 Hz band: every existing measured-IR dataset targets underwater
acoustic communications in the kHz range. KRAKEN/BELLHOP from canonical
environment descriptions remains the upstream path. As a Phase 1 shortcut,
C1.3-lite implements a parametric three-path model (direct + surface
bounce + bottom bounce) with geometry sampled from priors plus Thorpe
absorption.

Reflection conventions:
  - direct:  R_d = +1
  - surface: R_s = -1   (pressure-release boundary; Lloyd-mirror interference)
  - bottom:  R_b = 10^(-bottom_loss_db / 20)   (linear amplitude factor)

Per-path Thorpe absorption is applied before coherent summation, so each
path accumulates absorption proportional to its own slant range.
"""

from __future__ import annotations

import numpy as np

from fathom.synthetic.priors import SampledPropagationGeometry


PROPAGATION_MODEL_ID = "c1_3_lite_three_path_v1"


def thorpe_absorption_db_per_km(frequency_hz: np.ndarray | float) -> np.ndarray:
    """Thorpe (1967) volume absorption in dB/km.

    frequency_hz is Hz; converted to kHz internally for the standard formula:

        alpha(f) = 3.3e-3 + 0.11 * f^2 / (1 + f^2)
                         + 44   * f^2 / (4100 + f^2)
                         + 3e-4 * f^2        [f in kHz, alpha in dB/km]
    """
    f_khz = np.asarray(frequency_hz, dtype=np.float64) / 1000.0
    f2 = f_khz * f_khz
    return (
        3.3e-3
        + 0.11 * f2 / (1.0 + f2)
        + 44.0 * f2 / (4100.0 + f2)
        + 3.0e-4 * f2
    )


def _slant_ranges(geometry: SampledPropagationGeometry) -> tuple[float, float, float]:
    """Return (r_direct, r_surface, r_bottom) in meters."""
    h = geometry.horizontal_range_m
    z_s = geometry.source_depth_m
    z_r = geometry.receiver_depth_m
    H = geometry.water_depth_m
    r_d = float(np.sqrt(h * h + (z_s - z_r) ** 2))
    r_s = float(np.sqrt(h * h + (z_s + z_r) ** 2))
    r_b = float(np.sqrt(h * h + (2.0 * H - z_s - z_r) ** 2))
    return r_d, r_s, r_b


def three_path_response(
    frequency_hz: np.ndarray | float,
    geometry: SampledPropagationGeometry,
) -> np.ndarray:
    """Complex frequency response H(f) of the parametric three-path channel.

        H(f) = sum_i R_i * (1/r_i) * exp(-j 2 pi f r_i / c)
                          * 10^(-alpha(f) r_i / 20000)

    where the 20000 converts dB/km * m -> linear amplitude factor.

    Returns np.complex128 with the same shape as broadcast(frequency_hz).
    """
    f = np.asarray(frequency_hz, dtype=np.float64)
    c = float(geometry.sound_speed_m_per_s)
    r_d, r_s, r_b = _slant_ranges(geometry)
    R_d = 1.0
    R_s = -1.0  # pressure-release surface; Lloyd mirror
    R_b = 10.0 ** (-float(geometry.bottom_reflection_loss_db) / 20.0)

    alpha = thorpe_absorption_db_per_km(f)  # dB/km, same shape as f

    def _per_path(r: float, R: float) -> np.ndarray:
        absorption = 10.0 ** (-alpha * r / 20000.0)  # r in m, alpha in dB/km
        phase = np.exp(-1j * 2.0 * np.pi * f * r / c)
        return (R / r) * absorption * phase

    return _per_path(r_d, R_d) + _per_path(r_s, R_s) + _per_path(r_b, R_b)


def apply_three_path_channel(
    source_waveform: np.ndarray,
    sr: int,
    geometry: SampledPropagationGeometry,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the C1.3-lite three-path channel to a 1-D source waveform.

    rng is reserved for future stochastic effects (e.g., scattering jitter);
    the current model is deterministic given geometry.

    Returns (filtered_waveform, H_rfft, propagation_metadata).
      - filtered_waveform: same shape and dtype as source_waveform
      - H_rfft: complex frequency response at rFFT bins (np.fft.rfftfreq grid)
      - propagation_metadata: dict for the audit sidecar
    """
    del rng  # reserved
    if source_waveform.ndim != 1:
        raise ValueError(
            f"source_waveform must be 1-D; got shape {source_waveform.shape}"
        )
    n = source_waveform.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))

    H = three_path_response(freqs, geometry)

    X = np.fft.rfft(source_waveform.astype(np.float64))
    Y = X * H
    filtered = np.fft.irfft(Y, n=n).astype(source_waveform.dtype)

    r_d, r_s, r_b = _slant_ranges(geometry)
    c = geometry.sound_speed_m_per_s
    alpha_100 = float(thorpe_absorption_db_per_km(100.0))
    alpha_1k = float(thorpe_absorption_db_per_km(1000.0))

    def _path_meta(r: float, R_coeff: float) -> dict:
        return {
            "slant_range_m": float(r),
            "delay_s": float(r / c),
            "reflection_coefficient": float(R_coeff),
            "thorpe_loss_db_at_100hz": float(alpha_100 * r / 1000.0),
            "thorpe_loss_db_at_1khz": float(alpha_1k * r / 1000.0),
        }

    R_b_linear = 10.0 ** (-float(geometry.bottom_reflection_loss_db) / 20.0)
    metadata = {
        "model_id": PROPAGATION_MODEL_ID,
        "paths": {
            "direct": _path_meta(r_d, +1.0),
            "surface": _path_meta(r_s, -1.0),
            "bottom": _path_meta(r_b, R_b_linear),
        },
        "bottom_reflection_loss_db": float(geometry.bottom_reflection_loss_db),
    }

    return filtered, H, metadata