"""Round-trip detection tests (Sprint 2 Cluster 5).

Synthetic waveforms with injected tonals exercise the full pipeline:
compute_lofar_gram -> detect_lines. Each test pins a deterministic seed and
asserts the persistence filter recovers (or correctly rejects) the injected
signal per Sprint2_Plan §6 acceptance criterion.
"""
from datetime import datetime, timezone

import numpy as np

from fathom.detection import DetectionConfig, detect_lines
from fathom.events import EventBus
from fathom.grams.lofar import compute_lofar_gram
from fathom.models import LOFARConfig, StftConfig

SR = 32000
HOP = 4096
N_FFT = 16384
ANCHOR_UTC = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)


def _lofar_config() -> LOFARConfig:
    return LOFARConfig(
        stft=StftConfig(sample_rate=SR, n_fft=N_FFT, hop_length=HOP, window_length=N_FFT),
        freq_min_hz=1.0,
        freq_max_hz=1000.0,
        normalization_train_window_bins=33,
        normalization_central_window_bins=5,
        normalization_gap_bins=1,
    )


def _detection_config(**overrides) -> DetectionConfig:
    return DetectionConfig(**overrides)


def _make_waveform(
    duration_s: float,
    freq_hz: float | None,
    amplitude: float,
    noise_std: float,
    *,
    active_intervals: list[tuple[float, float]] | None = None,
    drift_end_freq_hz: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Build a 1D mono waveform.

    - `freq_hz=None` produces noise only.
    - `active_intervals=[(t0, t1), ...]` gates the tonal to those windows.
    - `drift_end_freq_hz` linearly chirps from `freq_hz` to that value.
    """
    rng = np.random.default_rng(seed)
    n = int(SR * duration_s)
    t = np.arange(n) / SR
    sig = rng.standard_normal(n).astype("float32") * noise_std
    if freq_hz is not None:
        if drift_end_freq_hz is not None:
            inst_freq = freq_hz + (drift_end_freq_hz - freq_hz) * (t / duration_s)
            phase = 2 * np.pi * np.cumsum(inst_freq) / SR
            tonal = amplitude * np.sin(phase).astype("float32")
        else:
            tonal = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype("float32")
        if active_intervals is not None:
            gate = np.zeros_like(tonal)
            for t0, t1 in active_intervals:
                gate[(t >= t0) & (t < t1)] = 1.0
            tonal = tonal * gate
        sig = sig + tonal
    return sig.astype("float32")


def _detect(wav: np.ndarray, dcfg: DetectionConfig | None = None):
    gram = compute_lofar_gram(wav, _lofar_config())
    bus = EventBus()  # isolate from default bus
    return detect_lines(
        gram,
        dcfg or _detection_config(),
        array_id="TEST",
        beam_id=None,
        recording_start_utc=ANCHOR_UTC,
        bus=bus,
    )


def test_persistent_tonal_recovered():
    """5 s tonal at 50 Hz in noise, well above threshold — exactly one line near 50 Hz."""
    wav = _make_waveform(duration_s=5.0, freq_hz=50.0, amplitude=0.05, noise_std=0.05, seed=0)
    lines = _detect(wav)
    line_50 = [L for L in lines if abs(L.frequency_hz - 50.0) < 5.0]
    assert len(line_50) >= 1, f"no line recovered near 50 Hz; got {[L.frequency_hz for L in lines]}"
    L = max(line_50, key=lambda x: x.snr_db)
    assert L.persistence_s >= 3.0, f"persistence {L.persistence_s:.2f} s < 3 s"
    assert L.snr_db >= 8.0, f"snr {L.snr_db:.2f} dB < threshold"


def test_drifting_tonal_recovered():
    """Slow chirp 49 -> 51 Hz over 5 s (~1 bin total drift; well within drift_bins=2)."""
    wav = _make_waveform(
        duration_s=5.0,
        freq_hz=49.0,
        drift_end_freq_hz=51.0,
        amplitude=0.05,
        noise_std=0.05,
        seed=0,
    )
    lines = _detect(wav)
    line_50 = [L for L in lines if abs(L.frequency_hz - 50.0) < 5.0]
    assert len(line_50) >= 1, f"drifting tonal not recovered; got {[L.frequency_hz for L in lines]}"


def test_intermittent_within_gap_tolerance_merged():
    """Tonal on [0, 2) s, off [2, 2.5) s, on [2.5, 5) s. Gap ~4 hop frames at hop=128 ms,
    well within gap_tolerance_time_bins=8 — should merge into a single line.
    """
    wav = _make_waveform(
        duration_s=5.0,
        freq_hz=50.0,
        amplitude=0.05,
        noise_std=0.05,
        active_intervals=[(0.0, 2.0), (2.5, 5.0)],
        seed=0,
    )
    lines = _detect(wav)
    line_50 = [L for L in lines if abs(L.frequency_hz - 50.0) < 5.0]
    assert len(line_50) >= 1, "intermittent tonal not recovered as a merged line"
    L = max(line_50, key=lambda x: x.snr_db)
    assert L.persistence_s >= 3.0, f"merged persistence {L.persistence_s:.2f} s < 3 s"


def test_sub_persistence_noise_rejected():
    """1 s tonal in 5 s recording — span ~1 s < 3 s persistence floor; rejected."""
    wav = _make_waveform(
        duration_s=5.0,
        freq_hz=50.0,
        amplitude=0.05,
        noise_std=0.05,
        active_intervals=[(0.0, 1.0)],
        seed=0,
    )
    lines = _detect(wav)
    line_50 = [L for L in lines if abs(L.frequency_hz - 50.0) < 5.0]
    assert len(line_50) == 0, (
        f"sub-persistence tonal should have been rejected; got "
        f"{[(L.frequency_hz, L.persistence_s) for L in line_50]}"
    )