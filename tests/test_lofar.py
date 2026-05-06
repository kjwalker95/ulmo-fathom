"""Smoke tests for LOFAR and DEMON gram computation."""
import numpy as np

from fathom.grams.lofar import compute_lofar_gram
from fathom.grams.demon import compute_demon_gram
from fathom.models import DEMONConfig, LOFARConfig, StftConfig


def _lofar_config() -> LOFARConfig:
    return LOFARConfig(
        stft=StftConfig(sample_rate=32000, n_fft=16384, hop_length=4096, window_length=16384),
        freq_min_hz=1.0,
        freq_max_hz=1000.0,
        normalization_train_window_bins=33,
        normalization_central_window_bins=5,
        normalization_gap_bins=1,
    )


def test_lofar_resolves_low_frequency_tone():
    sr = 32000
    duration_s = 4.0
    t = np.arange(int(sr * duration_s)) / sr
    # 50 Hz tone — analog of the auxiliary-machinery vulnerability band
    sig = (0.1 * np.sin(2 * np.pi * 50.0 * t)).astype("float32")
    sig += 0.01 * np.random.default_rng(0).standard_normal(sig.size).astype("float32")
    cfg = _lofar_config()
    gram = compute_lofar_gram(sig, cfg)
    assert gram.power_db.ndim == 2
    assert (gram.frequencies_hz >= 1.0).all() and (gram.frequencies_hz <= 1000.0).all()
    bin_idx = int(np.argmax(gram.normalized_power_db.mean(axis=1)))
    peak_hz = float(gram.frequencies_hz[bin_idx])
    assert abs(peak_hz - 50.0) < 5.0, f"peak at {peak_hz} Hz, expected ~50 Hz"


def test_demon_resolves_modulation():
    sr = 32000
    duration_s = 8.0  # post-decimation length must exceed n_fft (8s/100*32000 = 2560 >> 1024)
    t = np.arange(int(sr * duration_s)) / sr
    carrier = np.sin(2 * np.pi * 2000.0 * t)
    modulation = 1.0 + 0.5 * np.sin(2 * np.pi * 12.0 * t)  # 12 Hz envelope modulation
    sig = (carrier * modulation).astype("float32")
    cfg = DEMONConfig(
        sample_rate=sr,
        band_low_hz=500,
        band_high_hz=5000,
        envelope_lpf_cutoff_hz=150,
        decimation_factor=100,
        n_fft=1024,
        hop_length=256,
    )
    gram = compute_demon_gram(sig, cfg)
    assert gram.power_db.ndim == 2
    bin_idx = int(np.argmax(gram.power_db.mean(axis=1)))
    peak_hz = float(gram.frequencies_hz[bin_idx])
    assert abs(peak_hz - 12.0) < 5.0, f"DEMON peak at {peak_hz} Hz, expected ~12 Hz"