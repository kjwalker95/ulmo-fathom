"""Resampling round-trip tests (Sprint 3 Cluster 1)."""
import numpy as np
import pytest

from fathom.ingestion._resample import resample_to


def test_resample_no_op_when_rates_match():
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(32000).astype("float32")
    out = resample_to(wav, source_sr=32000, target_sr=32000)
    np.testing.assert_array_equal(out, wav)


def test_resample_preserves_50hz_tonal_shipsear_to_deepship_rate():
    """ShipsEar native 52,734 Hz -> Tuor's 32,000 Hz LOFAR rate.
    A 50 Hz tonal at the source rate should still peak at ~50 Hz after resampling.
    """
    source_sr = 52734
    target_sr = 32000
    duration_s = 4.0
    t = np.arange(int(source_sr * duration_s)) / source_sr
    sig = (0.5 * np.sin(2 * np.pi * 50.0 * t)).astype("float32")
    sig += 0.01 * np.random.default_rng(0).standard_normal(sig.size).astype("float32")

    out = resample_to(sig, source_sr=source_sr, target_sr=target_sr)

    # Output length should be approximately duration_s * target_sr (modulo polyphase filter delay).
    expected_len = int(duration_s * target_sr)
    assert abs(len(out) - expected_len) < 100, (
        f"expected length ~{expected_len}, got {len(out)}"
    )

    # Peak frequency in the resampled signal should be at ~50 Hz.
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), d=1.0 / target_sr)
    band = freqs < 1000  # restrict to the LOFAR primary view
    peak_freq = float(freqs[band][int(np.argmax(spectrum[band]))])
    assert abs(peak_freq - 50.0) < 1.0, f"peak at {peak_freq:.2f} Hz, expected ~50 Hz"


def test_resample_rejects_non_mono():
    rng = np.random.default_rng(0)
    stereo = rng.standard_normal((100, 2)).astype("float32")
    with pytest.raises(ValueError, match="mono"):
        resample_to(stereo, source_sr=44100, target_sr=32000)


def test_resample_rejects_invalid_rates():
    wav = np.zeros(100, dtype="float32")
    with pytest.raises(ValueError, match="positive"):
        resample_to(wav, source_sr=0, target_sr=32000)
    with pytest.raises(ValueError, match="positive"):
        resample_to(wav, source_sr=32000, target_sr=-1)