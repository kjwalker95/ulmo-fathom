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
from fathom.synthetic.priors import SampledTonalParameters

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

def _generate_pulse_onsets(
    rng: np.random.Generator,
    t_onset_s: float,
    total_persistence_s: float,
    cluster_period_s: float,
    pulses_per_cluster_range: tuple[int, int],
) -> list[float]:
    """A1 §3.3 cluster-timing model.

    Cluster centers are spaced by `cluster_period_s` with Gaussian jitter
    (sigma = 0.05 * T). Within each cluster, 1..N pulses arrive with Rayleigh
    jitter (sigma = 0.1 * T). Onsets outside [t_onset_s, t_onset_s +
    total_persistence_s] are dropped; remainder is sorted ascending.
    """
    if cluster_period_s <= 0:
        raise ValueError(f"cluster_period_s must be positive; got {cluster_period_s}")
    if total_persistence_s <= 0:
        return []

    n_clusters = max(1, int(np.ceil(total_persistence_s / cluster_period_s)))
    pmin, pmax = pulses_per_cluster_range
    if pmin < 1 or pmax < pmin:
        raise ValueError(f"invalid pulses_per_cluster_range {pulses_per_cluster_range}")

    onsets: list[float] = []
    persistence_end = t_onset_s + total_persistence_s

    for i in range(n_clusters):
        cluster_center_s = (
            t_onset_s
            + i * cluster_period_s
            + float(rng.normal(0.0, 0.05 * cluster_period_s))
        )
        n_pulses = int(rng.integers(pmin, pmax + 1))
        within_jitter = rng.rayleigh(scale=0.1 * cluster_period_s, size=n_pulses)
        for jitter in within_jitter:
            pulse_onset = cluster_center_s + float(jitter)
            if t_onset_s <= pulse_onset < persistence_end:
                onsets.append(pulse_onset)

    return sorted(onsets)


def _render_decaying_cosine_pulse(
    t_axis_s: np.ndarray,
    *,
    pulse_onset_s: float,
    source_onset_s: float,
    f0_hz: float,
    n_harmonics: int,
    harmonic_decay: float,
    decay_constant_per_s: float,
    drift_rate_hz_per_s: float,
    fundamental_amplitude: float,
    rng: np.random.Generator,
    sample_rate: int,
) -> np.ndarray:
    """One decaying-cosine pulse with linear drift across source lifetime.

    A1 §3.3: s(t) = sum_h a_h * exp(-gamma*(t-onset)) * cos(phi_h(t) + phi0_h)
    Phase is integrated analytically since f_h(t) is linear in t.
    Drift accumulates from `source_onset_s` (not pulse_onset) so phase remains
    consistent across the source's lifetime.
    """
    effective_duration_s = 9.2 / max(decay_constant_per_s, 1e-3)
    pulse_end_s = pulse_onset_s + effective_duration_s
    out = np.zeros_like(t_axis_s, dtype=np.float64)

    active = (t_axis_s >= pulse_onset_s) & (t_axis_s < pulse_end_s)
    if not active.any():
        return out.astype(np.float32)

    t_rel = t_axis_s[active] - pulse_onset_s  # 0..effective_duration
    tau_p = pulse_onset_s - source_onset_s   # offset from source onset
    envelope = np.exp(-decay_constant_per_s * t_rel)

    nyquist_hz = sample_rate / 2.0
    f_base_at_pulse_start = f0_hz + drift_rate_hz_per_s * tau_p

    for h in range(n_harmonics):
        harmonic_idx = h + 1  # h=0 => fundamental at 1*f0
        f_h_at_start = harmonic_idx * f_base_at_pulse_start
        if f_h_at_start <= 0 or f_h_at_start >= nyquist_hz:
            continue  # skip aliasing or non-physical harmonics

        # Closed-form phase: 2*pi*(h+1)*((f0+drift*tau_p)*t_rel + drift*t_rel^2/2)
        phase = (
            2.0 * np.pi * harmonic_idx
            * (f_base_at_pulse_start * t_rel + 0.5 * drift_rate_hz_per_s * t_rel ** 2)
        )
        phi0 = float(rng.uniform(0.0, 2.0 * np.pi))
        a_h = fundamental_amplitude * (harmonic_decay ** h)
        out[active] += a_h * envelope * np.cos(phase + phi0)

    return out.astype(np.float32)


def inject_parameterized_tonal(
    ambient: np.ndarray,
    sample_rate: int,
    *,
    params: SampledTonalParameters,
    rng: np.random.Generator,
    fade_s: float = 0.1,
    pulses_per_cluster_range: tuple[int, int] = (1, 5),
) -> tuple[np.ndarray, dict]:
    """Inject a parameterized multi-pulse tonal source into ambient.

    A1 §3.3 with C1.1 deltas. Returns (combined_signal, source_truth_dict).
    The truth dict carries per-source metadata (params snapshot, pulse onsets,
    fundamental amplitude, local RMS, harmonic table) consumed by C1.1.c
    (`compute_per_frame_truth`) to populate `SyntheticLineGroundTruth` rows.
    """
    if ambient.ndim != 1:
        raise ValueError(f"expected mono 1D ambient; got shape {ambient.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive; got {sample_rate}")
    if params.n_harmonics < 1:
        raise ValueError(f"params.n_harmonics must be >= 1; got {params.n_harmonics}")
    if not (0.0 < params.harmonic_decay <= 1.0):
        raise ValueError(
            f"params.harmonic_decay must be in (0, 1]; got {params.harmonic_decay}"
        )

    local_rms = _local_ambient_rms_at_frequency(ambient, sample_rate, params.f0_hz)
    if local_rms <= 0:
        raise ValueError(
            f"local ambient RMS at {params.f0_hz} Hz is zero; "
            "cannot compute SNR-relative tonal amplitude"
        )
    fundamental_amplitude = float(
        local_rms * (10.0 ** (params.target_snr_db / 20.0)) * np.sqrt(2)
    )

    pulse_onsets = _generate_pulse_onsets(
        rng,
        params.t_onset_s,
        params.total_persistence_s,
        params.cluster_period_s,
        pulses_per_cluster_range,
    )

    n = len(ambient)
    t_axis_s = np.arange(n) / sample_rate
    source_signal = np.zeros(n, dtype=np.float64)

    fade_samples_default = max(1, int(fade_s * sample_rate))
    effective_duration_s = 9.2 / max(params.decay_constant_per_s, 1e-3)

    for pulse_onset_s in pulse_onsets:
        pulse = _render_decaying_cosine_pulse(
            t_axis_s,
            pulse_onset_s=pulse_onset_s,
            source_onset_s=params.t_onset_s,
            f0_hz=params.f0_hz,
            n_harmonics=params.n_harmonics,
            harmonic_decay=params.harmonic_decay,
            decay_constant_per_s=params.decay_constant_per_s,
            drift_rate_hz_per_s=params.drift_rate_hz_per_s,
            fundamental_amplitude=fundamental_amplitude,
            rng=rng,
            sample_rate=sample_rate,
        )

        start_idx = max(0, int(round(pulse_onset_s * sample_rate)))
        end_idx = min(n, int(round((pulse_onset_s + effective_duration_s) * sample_rate)))
        if end_idx > start_idx:
            fade_samples = min(fade_samples_default, max(1, (end_idx - start_idx) // 2))
            gate = _cosine_taper_gate(n, start_idx, end_idx, fade_samples)
            pulse = (pulse * gate).astype(np.float32)

        source_signal += pulse

    combined = (ambient.astype(np.float64) + source_signal).astype(np.float32)

    harmonics_info: list[dict] = []
    for h in range(params.n_harmonics):
        h_freq = (h + 1) * params.f0_hz
        if h_freq >= sample_rate / 2:
            break
        amp = fundamental_amplitude * (params.harmonic_decay ** h)
        snr_db = (
            params.target_snr_db + 20.0 * h * np.log10(params.harmonic_decay)
            if params.harmonic_decay > 0 else params.target_snr_db
        )
        harmonics_info.append({
            "harmonic_id": h,
            "harmonic_freq_hz": float(h_freq),
            "amplitude": float(amp),
            "snr_db": float(snr_db),
        })

    truth = {
        "f0_hz": params.f0_hz,
        "n_harmonics": params.n_harmonics,
        "harmonic_decay": params.harmonic_decay,
        "decay_constant_per_s": params.decay_constant_per_s,
        "cluster_period_s": params.cluster_period_s,
        "total_persistence_s": params.total_persistence_s,
        "drift_rate_hz_per_s": params.drift_rate_hz_per_s,
        "target_snr_db": params.target_snr_db,
        "t_onset_s": params.t_onset_s,
        "pulse_onsets_s": pulse_onsets,
        "effective_duration_s": effective_duration_s,
        "fundamental_amplitude": fundamental_amplitude,
        "local_ambient_rms_fundamental": float(local_rms),
        "fade_s": fade_s,
        "harmonics": harmonics_info,
    }
    return combined, truth