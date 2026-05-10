"""Per-frame truth curves and mask-bin computation for C1.1 synthetic clips.

Populates SyntheticLineGroundTruth rows from the SampledTonalParameters +
pulse-onset metadata produced by inject_parameterized_tonal. Output curves
(freq_curve_hz, snr_curve_db, mask_bin_indices) are length = number of
active frames for that harmonic; t_start_s / t_end_s give the active bounds.

Frame-center timing follows scipy.signal.stft(boundary=None, padded=False) —
the convention used by fathom.grams.lofar.compute_lofar_gram. mask_bin_indices
align with that gram's (frame, freq) grid.
"""

from __future__ import annotations

import numpy as np

from fathom.models import StftConfig, SyntheticLineGroundTruth
from fathom.synthetic.tonals import _local_ambient_rms_at_frequency


def stft_frame_times_s(n_samples: int, stft: StftConfig) -> np.ndarray:
    """Frame-center times (seconds) matching scipy.signal.stft(boundary=None, padded=False).

    Returns an empty array if the signal is shorter than one window.
    """
    if n_samples < stft.window_length:
        return np.array([], dtype=float)
    n_frames = (n_samples - stft.window_length) // stft.hop_length + 1
    return (np.arange(n_frames) * stft.hop_length + stft.window_length / 2.0) / stft.sample_rate


def _compute_active_envelope(
    frame_times_s: np.ndarray,
    pulse_onsets_s: list[float],
    decay_constant_per_s: float,
    active_threshold_ratio: float = 1e-3,
) -> np.ndarray:
    """Per-frame envelope = max over overlapping pulses of exp(-gamma*(t - pulse_onset)).

    Frames with envelope below `active_threshold_ratio` are returned as zero —
    interpreted as inactive for masking purposes.
    """
    if frame_times_s.size == 0:
        return np.array([], dtype=float)
    log_thr = -np.log(max(active_threshold_ratio, 1e-12))
    active_duration_s = log_thr / max(decay_constant_per_s, 1e-3)

    envelope = np.zeros_like(frame_times_s)
    for pulse_onset_s in pulse_onsets_s:
        dt = frame_times_s - pulse_onset_s
        in_pulse = (dt >= 0) & (dt < active_duration_s)
        if not in_pulse.any():
            continue
        env_contrib = np.exp(-decay_constant_per_s * dt[in_pulse])
        envelope[in_pulse] = np.maximum(envelope[in_pulse], env_contrib)
    return envelope


def compute_per_frame_truth(
    source_truths: list[dict],
    source_ids: list[str],
    ambient: np.ndarray,
    stft: StftConfig,
    generation_seed: int,
) -> list[SyntheticLineGroundTruth]:
    """Build one SyntheticLineGroundTruth per (source, harmonic).

    For each harmonic of each source:
      - active frames = those where the source's max-overlapping pulse envelope
        is above 1e-3 of fundamental amplitude
      - freq_curve_hz: instantaneous (h+1)*(f0 + drift*(t_frame - t_onset)) at
        each active frame
      - snr_curve_db: 20*log10(harmonic_rms_at_frame / local_ambient_rms_h),
        where harmonic_rms_at_frame = a_h * envelope_at_frame / sqrt(2). The
        local ambient RMS is computed once per harmonic at its starting
        frequency (drift effect on ambient is small over typical clip durations)
      - mask_bin_indices: (frame_idx, bin_idx) tuples, bin_idx aligned to
        the LOFAR gram's STFT grid
    """
    if len(source_truths) != len(source_ids):
        raise ValueError(
            f"source_truths and source_ids length mismatch: "
            f"{len(source_truths)} vs {len(source_ids)}"
        )

    sample_rate = stft.sample_rate
    n_samples = len(ambient)
    frame_times_s = stft_frame_times_s(n_samples, stft)
    freq_resolution_hz = sample_rate / stft.n_fft
    nyquist_bin = stft.n_fft // 2 - 1
    frame_duration_s = stft.hop_length / sample_rate

    rows: list[SyntheticLineGroundTruth] = []

    for source_truth, source_id in zip(source_truths, source_ids):
        f0_hz = source_truth["f0_hz"]
        n_harmonics = source_truth["n_harmonics"]
        harmonic_decay = source_truth["harmonic_decay"]
        decay_constant = source_truth["decay_constant_per_s"]
        drift_rate = source_truth["drift_rate_hz_per_s"]
        t_onset_s = source_truth["t_onset_s"]
        fundamental_amplitude = source_truth["fundamental_amplitude"]
        pulse_onsets_s = source_truth["pulse_onsets_s"]

        envelope = _compute_active_envelope(frame_times_s, pulse_onsets_s, decay_constant)
        active_indices = np.where(envelope > 0)[0]

        for h in range(n_harmonics):
            h_freq_at_t_onset = (h + 1) * f0_hz
            if h_freq_at_t_onset >= sample_rate / 2:
                break  # harmonic above Nyquist; skip this and higher

            local_rms_h = _local_ambient_rms_at_frequency(
                ambient, sample_rate, h_freq_at_t_onset
            )
            if local_rms_h <= 0:
                # Pathological — skip this harmonic
                continue

            a_h = fundamental_amplitude * (harmonic_decay ** h)

            if active_indices.size == 0:
                rows.append(SyntheticLineGroundTruth(
                    line_id=f"{source_id}_h{h}",
                    source_id=source_id,
                    source_type="tonal",
                    harmonic_id=h,
                    f0_hz=float(f0_hz),
                    freq_curve_hz=[],
                    t_start_s=float(t_onset_s),
                    t_end_s=float(t_onset_s),
                    snr_curve_db=[],
                    persistence_s=0.0,
                    drift_rate_hz_per_s=float(drift_rate),
                    mask_bin_indices=[],
                    generation_seed=int(generation_seed),
                ))
                continue

            t_active = frame_times_s[active_indices]
            f_h_at_active = (h + 1) * (f0_hz + drift_rate * (t_active - t_onset_s))
            f_h_clamped = np.clip(f_h_at_active, 1e-3, sample_rate / 2.0 - 1.0)
            bin_indices = np.clip(
                np.round(f_h_clamped / freq_resolution_hz).astype(int),
                0,
                nyquist_bin,
            )

            envelope_active = envelope[active_indices]
            harmonic_rms_active = a_h * envelope_active / np.sqrt(2.0)
            snr_db_active = 20.0 * np.log10(
                np.maximum(harmonic_rms_active, 1e-12) / local_rms_h
            )

            t_start_s = float(t_active[0])
            t_end_s = float(t_active[-1] + frame_duration_s)

            rows.append(SyntheticLineGroundTruth(
                line_id=f"{source_id}_h{h}",
                source_id=source_id,
                source_type="tonal",
                harmonic_id=h,
                f0_hz=float(f0_hz),
                freq_curve_hz=[float(x) for x in f_h_at_active],
                t_start_s=t_start_s,
                t_end_s=t_end_s,
                snr_curve_db=[float(x) for x in snr_db_active],
                persistence_s=float(t_end_s - t_start_s),
                drift_rate_hz_per_s=float(drift_rate),
                mask_bin_indices=[
                    (int(fi), int(bi))
                    for fi, bi in zip(active_indices, bin_indices)
                ],
                generation_seed=int(generation_seed),
            ))

    return rows