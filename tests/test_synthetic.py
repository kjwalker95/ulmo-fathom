"""B1 + C1.1 synthetic generator tests."""
import numpy as np
import pytest
import soundfile as sf

from datetime import datetime, timezone

from fathom.models import (
    BiologicalClip,
    BiologicalClipLibrary,
    StftConfig,
    SyntheticConfuserLabel,
    SyntheticTruthManifest,
)
from fathom.synthetic import (
    BiologicalInjectionPriors,
    C1_1_GENERATOR_VERSION,
    C1_2_GENERATOR_VERSION,
    SampledTonalParameters,
    TonalParameterPriors,
    compute_per_frame_truth,
    generate_c1_1_clip,
    inject_biologicals,
    inject_deterministic_tonal,
    inject_parameterized_tonal,
    sample_tonal_parameters,
)
from fathom.synthetic.biologicals import sample_n_biologicals
from fathom.synthetic.tonals import (
    _generate_pulse_onsets,
    _render_decaying_cosine_pulse,
)

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

@pytest.fixture(scope="module")
def synthetic_ambient_path(tmp_path_factory):
    """A 60s synthetic ambient WAV at 32 kHz, used by orchestrator tests."""
    sr = 32000
    duration_s = 60.0
    rng = np.random.default_rng(20260510)
    audio = rng.normal(0, 0.01, int(sr * duration_s)).astype(np.float32)
    path = tmp_path_factory.mktemp("ambient") / "mock_ambient.wav"
    sf.write(str(path), audio, samplerate=sr, subtype="PCM_16")
    return path


def test_sample_tonal_parameters_respects_clip_duration():
    rng = np.random.default_rng(0)
    priors = TonalParameterPriors()
    for _ in range(100):
        params = sample_tonal_parameters(rng, priors, clip_duration_s=60.0)
        assert params is not None
        assert 0.0 <= params.t_onset_s
        assert params.t_onset_s + params.total_persistence_s <= 60.0 + 1e-6


def test_pulse_onsets_within_persistence_window():
    rng = np.random.default_rng(0)
    onsets = _generate_pulse_onsets(
        rng,
        t_onset_s=5.0,
        total_persistence_s=40.0,
        cluster_period_s=10.0,
        pulses_per_cluster_range=(1, 5),
    )
    assert len(onsets) > 0
    assert all(5.0 <= o < 45.0 for o in onsets)
    assert onsets == sorted(onsets)


def test_pulse_onsets_cluster_at_period():
    """Most period-buckets receive at least one pulse when 3 pulses/cluster fire deterministically."""
    rng = np.random.default_rng(0)
    period = 10.0
    onsets = _generate_pulse_onsets(
        rng,
        t_onset_s=0.0,
        total_persistence_s=1000.0,
        cluster_period_s=period,
        pulses_per_cluster_range=(3, 3),
    )
    n_buckets = 100
    buckets_with_pulses = len({int(o // period) for o in onsets if 0 <= o < n_buckets * period})
    assert buckets_with_pulses > int(0.7 * n_buckets), (
        f"only {buckets_with_pulses}/{n_buckets} buckets received a pulse; "
        "cluster timing too sparse"
    )


def test_decaying_cosine_envelope_decays():
    """Envelope at t=1s should be ~exp(-1) of envelope just after onset, for gamma=1.0."""
    rng = np.random.default_rng(0)
    sr = 32000
    t_axis = np.arange(int(20 * sr)) / sr
    pulse = _render_decaying_cosine_pulse(
        t_axis,
        pulse_onset_s=0.0,
        source_onset_s=0.0,
        f0_hz=100.0,
        n_harmonics=1,
        harmonic_decay=1.0,
        decay_constant_per_s=1.0,
        drift_rate_hz_per_s=0.0,
        fundamental_amplitude=1.0,
        rng=rng,
        sample_rate=sr,
    )

    def peak_in_window(sig, t_center, half_window=0.05):
        lo = int(max(0, (t_center - half_window) * sr))
        hi = int(min(len(sig), (t_center + half_window) * sr))
        return float(np.abs(sig[lo:hi]).max())

    ratio = peak_in_window(pulse, 1.0) / peak_in_window(pulse, 0.05)
    assert 0.30 < ratio < 0.45  # exp(-1) ≈ 0.368


def test_drift_produces_monotonic_freq_curve():
    """drift_rate=0.5 Hz/s ⇒ freq_curve_hz end-start delta ≈ 0.5 * persistence."""
    rng = np.random.default_rng(0)
    sr = 32000
    ambient = rng.normal(0, 0.01, sr * 60).astype(np.float32)
    stft = StftConfig(sample_rate=sr, n_fft=16384, hop_length=4096, window_length=16384)
    params = SampledTonalParameters(
        f0_hz=100.0,
        n_harmonics=1,
        harmonic_decay=1.0,
        decay_constant_per_s=0.05,  # slow decay so persistence fully populates
        cluster_period_s=60.0,
        total_persistence_s=40.0,
        drift_rate_hz_per_s=0.5,
        target_snr_db=15.0,
        t_onset_s=5.0,
    )
    _, source_truth = inject_parameterized_tonal(ambient, sr, params=params, rng=rng)
    rows = compute_per_frame_truth([source_truth], ["src_drift"], ambient, stft, generation_seed=0)
    fc = rows[0].freq_curve_hz
    assert len(fc) > 1
    assert fc[-1] > fc[0]
    expected_drift = 0.5 * (rows[0].t_end_s - rows[0].t_start_s - stft.hop_length / sr)
    assert abs((fc[-1] - fc[0]) - expected_drift) < 1.0


def test_mask_bin_indices_match_freq_curve():
    """For an undrifted f0=80 Hz tonal, mask_bin_indices map to the single nearest bin."""
    rng = np.random.default_rng(0)
    sr = 32000
    ambient = rng.normal(0, 0.01, sr * 60).astype(np.float32)
    stft = StftConfig(sample_rate=sr, n_fft=16384, hop_length=4096, window_length=16384)
    params = SampledTonalParameters(
        f0_hz=80.0,
        n_harmonics=1,
        harmonic_decay=1.0,
        decay_constant_per_s=0.1,
        cluster_period_s=20.0,
        total_persistence_s=30.0,
        drift_rate_hz_per_s=0.0,
        target_snr_db=15.0,
        t_onset_s=5.0,
    )
    _, source_truth = inject_parameterized_tonal(ambient, sr, params=params, rng=rng)
    rows = compute_per_frame_truth([source_truth], ["src_00"], ambient, stft, generation_seed=0)
    expected_bin = round(80.0 / (sr / stft.n_fft))
    actual_bins = {bi for _, bi in rows[0].mask_bin_indices}
    assert actual_bins == {expected_bin}


def test_per_frame_snr_curve_nonempty():
    """Each line has nonempty per-frame curves, all three of equal length."""
    rng = np.random.default_rng(0)
    sr = 32000
    ambient = rng.normal(0, 0.01, sr * 60).astype(np.float32)
    stft = StftConfig(sample_rate=sr, n_fft=16384, hop_length=4096, window_length=16384)
    params = SampledTonalParameters(
        f0_hz=100.0,
        n_harmonics=2,
        harmonic_decay=0.5,
        decay_constant_per_s=0.1,
        cluster_period_s=10.0,
        total_persistence_s=30.0,
        drift_rate_hz_per_s=0.0,
        target_snr_db=15.0,
        t_onset_s=5.0,
    )
    _, source_truth = inject_parameterized_tonal(ambient, sr, params=params, rng=rng)
    rows = compute_per_frame_truth([source_truth], ["src_00"], ambient, stft, generation_seed=0)
    assert len(rows) == 2  # 2 harmonics
    for row in rows:
        assert len(row.snr_curve_db) > 0
        assert len(row.snr_curve_db) == len(row.freq_curve_hz) == len(row.mask_bin_indices)


def test_inject_parameterized_round_trip():
    """Inject at known f0=120 Hz, recover dominant frequency in active interval within 2 Hz."""
    rng = np.random.default_rng(0)
    sr = 32000
    ambient = (rng.standard_normal(sr * 30) * 0.05).astype(np.float32)
    params = SampledTonalParameters(
        f0_hz=120.0,
        n_harmonics=1,
        harmonic_decay=1.0,
        decay_constant_per_s=0.05,
        cluster_period_s=60.0,
        total_persistence_s=20.0,
        drift_rate_hz_per_s=0.0,
        target_snr_db=20.0,
        t_onset_s=2.0,
    )
    combined, _ = inject_parameterized_tonal(ambient, sr, params=params, rng=rng)
    in_active = combined[int(2 * sr):int(22 * sr)]
    spectrum = np.abs(np.fft.rfft(in_active * np.hanning(len(in_active))))
    freqs = np.fft.rfftfreq(len(in_active), d=1.0 / sr)
    band = (freqs > 50.0) & (freqs < 200.0)
    peak_freq = float(freqs[band][int(np.argmax(spectrum[band]))])
    assert abs(peak_freq - 120.0) < 2.0


def test_generate_c1_1_clip_writes_triplet(synthetic_ambient_path, tmp_path):
    """End-to-end: WAV + truth manifest JSON + audit sidecar all written and validate."""
    out_path = tmp_path / "clip.wav"
    result = generate_c1_1_clip(
        ambient_path=synthetic_ambient_path,
        out_path=out_path,
        seed=42,
    )
    assert result["wav_path"].exists()
    assert result["manifest_path"].exists()
    assert result["audit_path"].exists()
    manifest = SyntheticTruthManifest.model_validate_json(result["manifest_path"].read_text())
    assert manifest.generator_version == C1_1_GENERATOR_VERSION
    assert manifest.clip_id == "clip"


def test_negative_clip_has_no_lines(synthetic_ambient_path, tmp_path):
    """n_sources_distribution={0: 1.0} ⇒ negative_label=True, lines=[], audio == ambient."""
    out_path = tmp_path / "negative.wav"
    result = generate_c1_1_clip(
        ambient_path=synthetic_ambient_path,
        out_path=out_path,
        seed=1,
        priors=TonalParameterPriors(n_sources_distribution={0: 1.0}),
    )
    assert result["negative_label"] is True
    assert result["n_sources_realized"] == 0
    manifest = SyntheticTruthManifest.model_validate_json(result["manifest_path"].read_text())
    assert manifest.negative_label is True
    assert manifest.lines == []
    written, _ = sf.read(str(result["wav_path"]))
    original, _ = sf.read(str(synthetic_ambient_path))
    np.testing.assert_allclose(
        written.astype(np.float32), original.astype(np.float32), atol=1e-3
    )


def test_min_freq_separation_enforced(synthetic_ambient_path, tmp_path):
    """All pairwise f0 distances ≥ min_freq_separation_hz across forced-3-source clip."""
    out_path = tmp_path / "three_sources.wav"
    result = generate_c1_1_clip(
        ambient_path=synthetic_ambient_path,
        out_path=out_path,
        seed=42,
        priors=TonalParameterPriors(
            n_sources_distribution={3: 1.0},
            min_freq_separation_hz=20.0,
        ),
    )
    manifest = SyntheticTruthManifest.model_validate_json(result["manifest_path"].read_text())
    f0s = sorted({l.f0_hz for l in manifest.lines})
    assert len(f0s) >= 1
    for i in range(len(f0s)):
        for j in range(i + 1, len(f0s)):
            assert abs(f0s[j] - f0s[i]) >= 20.0, (
                f"f0 separation violated: {f0s[i]} vs {f0s[j]} "
                f"(delta={abs(f0s[j] - f0s[i]):.2f} Hz)"
            )


def test_source_id_groups_harmonics(synthetic_ambient_path, tmp_path):
    """n_sources=2 with n_harmonics=3 ⇒ exactly 2 source_ids each appearing 3 times."""
    out_path = tmp_path / "two_three.wav"
    result = generate_c1_1_clip(
        ambient_path=synthetic_ambient_path,
        out_path=out_path,
        seed=42,
        priors=TonalParameterPriors(
            n_sources_distribution={2: 1.0},
            n_harmonics_choices=(3,),
        ),
    )
    manifest = SyntheticTruthManifest.model_validate_json(result["manifest_path"].read_text())
    if result["n_sources_realized"] == 2:
        distinct_source_ids = {l.source_id for l in manifest.lines}
        assert len(distinct_source_ids) == 2
        for sid in distinct_source_ids:
            srows = [l for l in manifest.lines if l.source_id == sid]
            assert len(srows) == 3
            assert sorted(l.harmonic_id for l in srows) == [0, 1, 2]




# ===========================================================================
# C1.2: biological confuser injection (A1 §3.2 + DCLDE 2018 source)
# ===========================================================================


@pytest.fixture(scope="module")
def biological_library_path(tmp_path_factory):
    """Fake 3-clip biological library — 2 Bm + 1 Eg — for self-contained tests."""
    root = tmp_path_factory.mktemp("bio_library")
    sr = 2000  # match DCLDE native LF rate
    rng = np.random.default_rng(1)
    clips: list[BiologicalClip] = []

    for i, (species_code, species_name, freq_range) in enumerate([
        ("Bm", "blue_whale", (10.0, 30.0)),
        ("Bm", "blue_whale", (10.0, 30.0)),
        ("Eg", "north_atlantic_right_whale", (50.0, 200.0)),
    ]):
        site_dir = root / species_code / "TEST"
        site_dir.mkdir(parents=True, exist_ok=True)
        clip_id = f"TEST_TEST_{species_code}_{i:05d}"
        audio = rng.normal(0, 0.05, sr * 5).astype(np.float32)
        clip_path = site_dir / f"{clip_id}.wav"
        sf.write(str(clip_path), audio, samplerate=sr, subtype="PCM_16")
        clips.append(BiologicalClip(
            clip_id=clip_id,
            source_dataset="test",
            species_code=species_code,
            species_name=species_name,
            site="TEST",
            deployment="test_dep_01",
            sample_rate_hz=sr,
            duration_s=5.0,
            pad_s=0.5,
            annotated_t_start_s=0.5,
            annotated_t_end_s=4.5,
            freq_range_hz=freq_range,
            quality="good",
            sha256="0" * 64,
            relative_path=str(clip_path.relative_to(root)),
        ))

    library = BiologicalClipLibrary(
        library_id="test_library_v1",
        source_dataset="test",
        n_clips=len(clips),
        species_counts={"Bm": 2, "Eg": 1},
        clips=clips,
        built_at=datetime.now(timezone.utc),
    )
    (root / "manifest.json").write_text(library.model_dump_json(indent=2))
    return root


def test_biological_priors_validate_distribution():
    """Distribution that doesn't sum to 1 must raise."""
    with pytest.raises(ValueError, match="must sum to 1"):
        BiologicalInjectionPriors(n_biologicals_distribution={0: 0.5, 1: 0.4})


def test_sample_n_biologicals_matches_distribution():
    """Empirical frequency over 5k draws lands within 2pp of priors."""
    rng = np.random.default_rng(0)
    priors = BiologicalInjectionPriors(
        n_biologicals_distribution={0: 0.40, 1: 0.30, 2: 0.20, 3: 0.10}
    )
    counts = np.bincount(
        [sample_n_biologicals(rng, priors) for _ in range(5000)], minlength=4
    )
    freqs = counts / counts.sum()
    assert abs(freqs[0] - 0.40) < 0.02
    assert abs(freqs[1] - 0.30) < 0.02
    assert abs(freqs[2] - 0.20) < 0.02
    assert abs(freqs[3] - 0.10) < 0.02


def test_inject_biologicals_zero_path(biological_library_path):
    """n=0 priors ⇒ combined == ambient, no overlays returned."""
    from fathom.synthetic.biologicals import load_biological_library

    rng = np.random.default_rng(0)
    sr = 32000
    ambient = rng.normal(0, 0.01, sr * 10).astype(np.float32)
    library = load_biological_library(biological_library_path)
    priors = BiologicalInjectionPriors(n_biologicals_distribution={0: 1.0})

    combined, overlays = inject_biologicals(
        ambient, sr,
        library=library, library_root=biological_library_path,
        priors=priors, rng=rng,
    )
    assert overlays == []
    np.testing.assert_array_equal(combined, ambient.astype(np.float32))


def test_inject_biologicals_species_weights_force_eg(biological_library_path):
    """species_weights={'Eg': 1.0} ⇒ all overlays are Eg even though Bm dominates the library."""
    from fathom.synthetic.biologicals import load_biological_library

    rng = np.random.default_rng(0)
    sr = 32000
    ambient = rng.normal(0, 0.01, sr * 30).astype(np.float32)
    library = load_biological_library(biological_library_path)
    priors = BiologicalInjectionPriors(
        n_biologicals_distribution={3: 1.0},
        species_weights={"Eg": 1.0},
    )

    combined, overlays = inject_biologicals(
        ambient, sr,
        library=library, library_root=biological_library_path,
        priors=priors, rng=rng,
    )
    assert len(overlays) == 3
    assert all(ov.species_code == "Eg" for ov in overlays)
    # Audio differs from ambient since overlays injected real energy
    diff_rms = float(np.sqrt(np.mean((combined - ambient) ** 2)))
    assert diff_rms > 1e-6


def test_generate_c1_1_clip_with_biologicals_populates_confusers(
    synthetic_ambient_path, biological_library_path, tmp_path,
):
    """biological_library_root set ⇒ generator_version=C1_2, confuser_labels populated."""
    result = generate_c1_1_clip(
        ambient_path=synthetic_ambient_path,
        out_path=tmp_path / "with_bio.wav",
        seed=42,
        biological_library_root=biological_library_path,
        biological_priors=BiologicalInjectionPriors(
            n_biologicals_distribution={2: 1.0},
        ),
    )
    assert result["biologicals_enabled"] is True
    assert result["n_biologicals_realized"] == 2

    manifest = SyntheticTruthManifest.model_validate_json(
        result["manifest_path"].read_text()
    )
    assert manifest.generator_version == C1_2_GENERATOR_VERSION
    assert len(manifest.confuser_labels) == 2
    for cl in manifest.confuser_labels:
        assert isinstance(cl, SyntheticConfuserLabel)
        assert cl.confuser_clip_id  # field rename: not watkins_id
        assert cl.source_dataset == "test"
        assert cl.target_snr_db is not None


def test_generate_c1_1_clip_negative_tonal_with_biologicals(
    synthetic_ambient_path, biological_library_path, tmp_path,
):
    """n_sources=0 with biologicals ⇒ negative_label=True, lines=[], confusers nonempty."""
    result = generate_c1_1_clip(
        ambient_path=synthetic_ambient_path,
        out_path=tmp_path / "neg_with_bio.wav",
        seed=1,
        priors=TonalParameterPriors(n_sources_distribution={0: 1.0}),
        biological_library_root=biological_library_path,
        biological_priors=BiologicalInjectionPriors(
            n_biologicals_distribution={2: 1.0},
        ),
    )
    assert result["negative_label"] is True
    assert result["n_sources_realized"] == 0
    assert result["n_biologicals_realized"] == 2

    manifest = SyntheticTruthManifest.model_validate_json(
        result["manifest_path"].read_text()
    )
    assert manifest.negative_label is True  # negative tracks TONALS only
    assert manifest.lines == []
    assert len(manifest.confuser_labels) == 2