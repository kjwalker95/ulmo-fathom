"""Deterministic tonal injection.

B1 minimum viable per A1 §7 item 13 + 2026-05-10 operator-review extensions:
  - Local-bin SNR calculation (per A1 §3.3 "SNR computed against local ambient
    at tonal frequency"; matches PCD v3 §6.3 classical-detector convention)
  - 1-3 harmonics with amplitude decay (per A1 §3.3 default n_harmonics=3,
    decay 0.3-0.7)
  - Cosine fade-in/fade-out on gate edges (suppresses broadband onset/offset
    transients)

C1 expands to the full A1 §3.3 parameterized schema with decaying-cosine
pulses + Rayleigh-jittered cluster timing + drift + propagation IRs.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _bandpass(
    waveform: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """4th-order Butterworth bandpass; sosfiltfilt for zero-phase."""
    nyq = sample_rate / 2.0
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 1.0 - 1e-6)
    if low >= high:
        raise ValueError(f"invalid bandpass [{low_hz}, {high_hz}] for nyq={nyq}")
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, waveform).astype(waveform.dtype, copy=False)


def _local_ambient_rms_at_frequency(
    ambient: np.ndarray,
    sample_rate: int,
    frequency_hz: float,
    *,
    bandwidth_factor: float = 0.1,
    min_bandwidth_hz: float = 4.0,
) -> float:
    """RMS of ambient bandpassed around target frequency (±bandwidth)."""
    bw = max(bandwidth_factor * frequency_hz, min_bandwidth_hz)
    low = max(frequency_hz - bw, 1.0)
    high = min(frequency_hz + bw, sample_rate / 2.0 - 1.0)
    band = _bandpass(ambient, sample_rate, low, high)
    return float(np.sqrt(np.mean(band ** 2)))


def _cosine_taper_gate(
    n: int,
    start_idx: int,
    end_idx: int,
    fade_samples: int,
) -> np.ndarray:
    """Build a gate that's 0 outside [start_idx, end_idx], 1 inside body, with
    cosine fade-in/out at edges. fade_samples is duration of each fade region."""
    gate = np.zeros(n, dtype="float32")
    fade_samples = max(1, fade_samples)
    body_end = min(end_idx, n)
    if start_idx >= body_end:
        return gate

    gate[start_idx:body_end] = 1.0  # body of gate

    fade_in_end = min(start_idx + fade_samples, body_end)
    if fade_in_end > start_idx:
        idx = np.arange(start_idx, fade_in_end)
        phase = (idx - start_idx) / fade_samples
        gate[idx] = 0.5 * (1.0 - np.cos(np.pi * phase))

    fade_out_start = max(body_end - fade_samples, start_idx)
    if body_end > fade_out_start:
        idx = np.arange(fade_out_start, body_end)
        phase = (body_end - idx) / fade_samples
        gate[idx] = 0.5 * (1.0 - np.cos(np.pi * phase))

    return gate


def inject_deterministic_tonal(
    ambient: np.ndarray,
    sample_rate: int,
    *,
    frequency_hz: float,
    t_start_s: float,
    t_end_s: float,
    target_snr_db: float,
    n_harmonics: int = 3,
    harmonic_amplitude_decay: float = 0.7,
    fade_s: float = 0.2,
) -> tuple[np.ndarray, dict]:
    """Inject a sinusoid (with optional harmonics) into ambient.

    SNR is computed against LOCAL ambient at the fundamental frequency (per
    A1 §3.3; matches PCD v3 §6.3 classical-detector convention). Harmonics
    decay exponentially per A1 §3.3 (default 0.7 → fundamental at +SNR,
    harmonic-1 at +SNR-3 dB, harmonic-2 at +SNR-6 dB).

    Gate edges have cosine fade-in/fade-out (default 200 ms) to suppress
    broadband onset/offset transients.

    Returns (combined_waveform, ground_truth_dict). ground_truth["harmonics"]
    is a list of per-harmonic dicts with `harmonic_id`, `harmonic_freq_hz`,
    `amplitude`, `snr_db`.
    """
    if ambient.ndim != 1:
        raise ValueError(f"expected mono 1D ambient; got shape {ambient.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive; got {sample_rate}")
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be >= 1; got {n_harmonics}")
    if not (0.0 < harmonic_amplitude_decay <= 1.0):
        raise ValueError(f"harmonic_amplitude_decay must be in (0, 1]; got {harmonic_amplitude_decay}")

    n = len(ambient)
    duration_s = n / sample_rate
    if t_start_s < 0 or t_end_s > duration_s or t_start_s >= t_end_s:
        raise ValueError(
            f"tonal time window [{t_start_s}, {t_end_s}] invalid for "
            f"ambient duration {duration_s:.2f}s"
        )

    local_rms = _local_ambient_rms_at_frequency(ambient, sample_rate, frequency_hz)
    if local_rms <= 0:
        raise ValueError(
            f"local ambient RMS at {frequency_hz} Hz is zero; cannot compute "
            "SNR-relative tonal amplitude"
        )

    # Tonal RMS = local_rms * 10^(snr/20); A = RMS * sqrt(2)
    fundamental_amplitude = float(local_rms * (10 ** (target_snr_db / 20)) * np.sqrt(2))

    t = np.arange(n) / sample_rate
    tonal = np.zeros(n, dtype="float32")
    harmonics_info = []

    for h_idx in range(n_harmonics):
        harmonic_freq = frequency_hz * (h_idx + 1)
        if harmonic_freq >= sample_rate / 2:
            break
        amp = fundamental_amplitude * (harmonic_amplitude_decay ** h_idx)
        tonal += (amp * np.sin(2 * np.pi * harmonic_freq * t)).astype("float32")
        harmonic_snr = float(target_snr_db + 20.0 * h_idx * np.log10(harmonic_amplitude_decay))
        harmonics_info.append({
            "harmonic_id": h_idx,
            "harmonic_freq_hz": float(harmonic_freq),
            "amplitude": float(amp),
            "snr_db": harmonic_snr,
        })

    start_idx = int(t_start_s * sample_rate)
    end_idx = int(t_end_s * sample_rate)
    fade_samples = int(fade_s * sample_rate)
    gate = _cosine_taper_gate(n, start_idx, end_idx, fade_samples)
    tonal *= gate

    combined = (ambient + tonal).astype("float32")

    ground_truth = {
        "frequency_hz": float(frequency_hz),
        "t_start_s": float(t_start_s),
        "t_end_s": float(t_end_s),
        "persistence_s": float(t_end_s - t_start_s),
        "target_snr_db": float(target_snr_db),
        "fundamental_amplitude": fundamental_amplitude,
        "local_ambient_rms_fundamental": local_rms,
        "n_harmonics_injected": len(harmonics_info),
        "harmonic_amplitude_decay": float(harmonic_amplitude_decay),
        "fade_s": float(fade_s),
        "harmonics": harmonics_info,
    }
    return combined, ground_truth