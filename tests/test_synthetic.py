"""B1 synthetic generator smoke tests."""
import numpy as np
import pytest

from fathom.synthetic.tonals import inject_deterministic_tonal


def test_tonal_injection_preserves_ambient_outside_gate():
    rng = np.random.default_rng(0)
    sr = 32000
    ambient = rng.standard_normal(sr * 5).astype("float32")
    combined, gt = inject_deterministic_tonal(
        ambient, sample_rate=sr,
        frequency_hz=50.0, t_start_s=1.0, t_end_s=4.0, target_snr_db=10.0,
    )
    np.testing.assert_array_equal(combined[:sr], ambient[:sr])         # pre-gate
    np.testing.assert_array_equal(combined[sr * 4:], ambient[sr * 4:])  # post-gate


def test_tonal_injection_inside_gate_has_tonal_energy():
    rng = np.random.default_rng(0)
    sr = 32000
    ambient = (rng.standard_normal(sr * 5) * 0.05).astype("float32")
    combined, _ = inject_deterministic_tonal(
        ambient, sample_rate=sr,
        frequency_hz=50.0, t_start_s=1.0, t_end_s=4.0, target_snr_db=20.0,
    )
    in_gate = combined[sr:sr * 4]
    spectrum = np.abs(np.fft.rfft(in_gate))
    freqs = np.fft.rfftfreq(len(in_gate), d=1.0 / sr)
    band = freqs < 1000
    peak_freq = float(freqs[band][int(np.argmax(spectrum[band]))])
    assert abs(peak_freq - 50.0) < 1.0


def test_tonal_injection_rejects_out_of_window():
    ambient = np.ones(32000 * 3, dtype="float32") * 0.1
    with pytest.raises(ValueError, match="invalid"):
        inject_deterministic_tonal(
            ambient, sample_rate=32000,
            frequency_hz=50.0, t_start_s=2.0, t_end_s=5.0, target_snr_db=10.0,
        )


def test_tonal_injection_rejects_silent_ambient():
    ambient = np.zeros(32000 * 3, dtype="float32")
    with pytest.raises(ValueError, match="silent"):
        inject_deterministic_tonal(
            ambient, sample_rate=32000,
            frequency_hz=50.0, t_start_s=1.0, t_end_s=2.0, target_snr_db=10.0,
        )