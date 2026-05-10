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
    with pytest.raises(ValueError, match="zero"):
        inject_deterministic_tonal(
            ambient, sample_rate=32000,
            frequency_hz=50.0, t_start_s=1.0, t_end_s=2.0, target_snr_db=10.0,
        )

def test_tonal_injection_includes_harmonics():
    """With n_harmonics=3, ground truth carries 3 harmonic entries."""
    rng = np.random.default_rng(0)
    sr = 32000
    ambient = (rng.standard_normal(sr * 5) * 0.05).astype("float32")
    _, gt = inject_deterministic_tonal(
        ambient, sample_rate=sr,
        frequency_hz=50.0, t_start_s=1.0, t_end_s=4.0, target_snr_db=10.0,
        n_harmonics=3, harmonic_amplitude_decay=0.7,
    )
    assert gt["n_harmonics_injected"] == 3
    harmonics = gt["harmonics"]
    assert harmonics[0]["harmonic_freq_hz"] == 50.0
    assert harmonics[1]["harmonic_freq_hz"] == 100.0
    assert harmonics[2]["harmonic_freq_hz"] == 150.0
    # Decay: harmonic-1 SNR = fundamental SNR + 20*log10(0.7) ≈ -3 dB
    assert abs(harmonics[1]["snr_db"] - (10.0 + 20 * np.log10(0.7))) < 1e-3
    # Decay: harmonic-2 SNR = +20*log10(0.7^2) ≈ -6 dB
    assert abs(harmonics[2]["snr_db"] - (10.0 + 40 * np.log10(0.7))) < 1e-3


def test_tonal_injection_local_snr_not_global_rms():
    """SNR should be relative to local-bin ambient RMS, not global RMS."""
    sr = 32000
    # Construct ambient with strong low-freq content + quiet high-freq
    rng = np.random.default_rng(0)
    n = sr * 5
    t = np.arange(n) / sr
    low_freq_strong = (0.5 * np.sin(2 * np.pi * 5.0 * t)).astype("float32")  # strong at 5 Hz
    bg_noise = (0.01 * rng.standard_normal(n)).astype("float32")
    ambient = (low_freq_strong + bg_noise).astype("float32")
    # Inject at 200 Hz where local ambient is just bg noise (quiet)
    combined_quiet, gt_quiet = inject_deterministic_tonal(
        ambient, sample_rate=sr,
        frequency_hz=200.0, t_start_s=1.0, t_end_s=4.0, target_snr_db=10.0,
        n_harmonics=1,
    )
    # The fundamental amplitude should reflect QUIET local ambient at 200 Hz,
    # not the loud global RMS dominated by the 5 Hz tonal.
    # Specifically, fundamental amplitude must be MUCH smaller than what global-RMS
    # calculation would have produced.
    global_rms = float(np.sqrt(np.mean(ambient ** 2)))
    naive_global_amplitude = global_rms * (10 ** (10.0 / 20)) * np.sqrt(2)
    assert gt_quiet["fundamental_amplitude"] < naive_global_amplitude * 0.1, (
        f"fundamental_amplitude={gt_quiet['fundamental_amplitude']:.4f} "
        f"should be << naive global={naive_global_amplitude:.4f} "
        "(local-bin SNR at quiet 200 Hz should give much smaller amplitude)"
    )


def test_cosine_fade_gate_smooths_edges():
    """Gate should rise smoothly from 0 (rather than hard step at t_start)."""
    rng = np.random.default_rng(0)
    sr = 32000
    ambient = (rng.standard_normal(sr * 5) * 0.05).astype("float32")
    fade_s = 0.2
    t_start = 1.0
    combined, _ = inject_deterministic_tonal(
        ambient, sample_rate=sr,
        frequency_hz=50.0, t_start_s=t_start, t_end_s=4.0,
        target_snr_db=20.0, n_harmonics=1, fade_s=fade_s,
    )
    # At t_start exactly, gate=0 → combined ≈ ambient
    np.testing.assert_allclose(
        combined[int(sr * t_start)], ambient[int(sr * t_start)], atol=1e-6
    )
    # Mid-fade (halfway through fade-in): gate ≈ 0.5
    mid_fade_idx = int(sr * (t_start + fade_s / 2))
    # Combined - ambient ≈ tonal at gate=0.5; should be much smaller than at full gate
    full_gate_idx = int(sr * (t_start + fade_s + 0.1))
    diff_mid = abs(combined[mid_fade_idx] - ambient[mid_fade_idx])
    diff_full = abs(combined[full_gate_idx] - ambient[full_gate_idx])
    # At mid-fade tonal is roughly 0.5x amplitude vs full gate; allow tolerance
    # (sinusoid value at one specific sample varies; check a small window mean)
    window = 10
    diff_mid_window = float(np.mean(np.abs(
        combined[mid_fade_idx - window:mid_fade_idx + window] -
        ambient[mid_fade_idx - window:mid_fade_idx + window]
    )))
    diff_full_window = float(np.mean(np.abs(
        combined[full_gate_idx - window:full_gate_idx + window] -
        ambient[full_gate_idx - window:full_gate_idx + window]
    )))
    assert diff_mid_window < diff_full_window