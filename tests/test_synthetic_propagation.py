"""Tests for the C1.3-lite propagation module and generator wiring.

Covers all four team-review-2026-05-12 regression items:
  Issue 1: boost-then-propagate preserves received SNR at f0.
  Issue 2: direct path uses Pythagorean slant range, not horizontal range.
  Issue 3: per-path Thorpe absorption (each path's absorption proportional
           to its own slant range).
  Issue 4: surface reflection coefficient = -1 (Lloyd mirror; pressure-release).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fathom.synthetic.generator import (
    C1_3_GENERATOR_VERSION,
    C1_3_LITE_DELTAS,
    generate_c1_1_clip,
)
from fathom.synthetic.priors import (
    PropagationGeometryPriors,
    SampledPropagationGeometry,
    TonalParameterPriors,
    sample_propagation_geometry,
)
from fathom.synthetic.propagation import (
    PROPAGATION_MODEL_ID,
    _slant_ranges,
    apply_three_path_channel,
    thorpe_absorption_db_per_km,
    three_path_response,
)


# ---------- Fixtures + helpers ----------

@pytest.fixture(scope="module")
def synthetic_ambient_path(tmp_path_factory):
    """60 s of synthetic ambient at 32 kHz, mirrored from test_synthetic.py."""
    sr = 32000
    duration_s = 60.0
    rng = np.random.default_rng(20260510)
    audio = rng.normal(0, 0.01, int(sr * duration_s)).astype(np.float32)
    path = tmp_path_factory.mktemp("ambient") / "mock_ambient.wav"
    sf.write(str(path), audio, samplerate=sr, subtype="PCM_16")
    return path


def _geom(**overrides) -> SampledPropagationGeometry:
    defaults = dict(
        water_depth_m=200.0,
        source_depth_m=5.0,
        receiver_depth_m=50.0,
        horizontal_range_m=5_000.0,
        sound_speed_m_per_s=1500.0,
        bottom_reflection_loss_db=6.0,
    )
    defaults.update(overrides)
    return SampledPropagationGeometry(**defaults)


# ---------- Thorpe absorption ----------

def test_thorpe_absorption_ratio_above_10x():
    a100 = float(thorpe_absorption_db_per_km(100.0))
    a1k = float(thorpe_absorption_db_per_km(1000.0))
    assert a1k / a100 > 10.0, f"Thorpe ratio must exceed 10x; got {a1k / a100:.2f}"


def test_thorpe_absorption_monotonic_through_band():
    freqs = np.logspace(1, 4, 30)  # 10 Hz → 10 kHz
    alpha = thorpe_absorption_db_per_km(freqs)
    diffs = np.diff(alpha)
    assert np.all(diffs > 0), "Thorpe absorption must be strictly increasing with frequency"


# ---------- Geometry sampling ----------

def test_geometry_sampling_respects_water_depth():
    rng = np.random.default_rng(0)
    priors = PropagationGeometryPriors()
    for _ in range(2000):
        geo = sample_propagation_geometry(rng, priors)
        assert geo is not None
        assert geo.source_depth_m <= geo.water_depth_m
        assert geo.receiver_depth_m <= geo.water_depth_m


def test_geometry_sampling_respects_prior_ranges():
    rng = np.random.default_rng(1)
    priors = PropagationGeometryPriors()
    for _ in range(500):
        geo = sample_propagation_geometry(rng, priors)
        assert geo is not None
        assert priors.water_depth_m_range[0] <= geo.water_depth_m <= priors.water_depth_m_range[1]
        assert priors.horizontal_range_m_range[0] <= geo.horizontal_range_m <= priors.horizontal_range_m_range[1]
        assert priors.bottom_reflection_loss_db_range[0] <= geo.bottom_reflection_loss_db <= priors.bottom_reflection_loss_db_range[1]


# ---------- Issue 2: direct-path slant range ----------

def test_direct_path_uses_slant_range_not_horizontal():
    """Issue 2 regression. Pythagorean direct-path geometry must include the
    source/receiver depth delta (not just the horizontal range)."""
    geo = _geom(source_depth_m=5.0, receiver_depth_m=500.0, horizontal_range_m=5_000.0)
    r_d, _r_s, _r_b = _slant_ranges(geo)
    expected = float(np.sqrt(5000.0**2 + (5.0 - 500.0) ** 2))
    assert abs(r_d - expected) < 1e-6
    assert r_d > 5000.0, "slant range must exceed horizontal range when depths differ"


# ---------- Issue 3: per-path Thorpe ----------

def test_per_path_absorption_differs_by_distance():
    """Issue 3 regression. The longest path (bottom bounce here) must
    accumulate more HF absorption than the direct path under the per-path
    Thorpe model."""
    src = np.zeros(32000, dtype=np.float32)
    src[0] = 1.0  # impulse — content irrelevant; we only need the metadata
    _y, _H, meta = apply_three_path_channel(
        src, 32000, _geom(), np.random.default_rng(0)
    )
    direct_hf = meta["paths"]["direct"]["thorpe_loss_db_at_1khz"]
    bottom_hf = meta["paths"]["bottom"]["thorpe_loss_db_at_1khz"]
    assert bottom_hf > direct_hf, (
        f"bottom HF loss ({bottom_hf:.6f}) must exceed direct HF loss "
        f"({direct_hf:.6f}) for the same geometry"
    )


# ---------- Issue 4: surface reflection coefficient + Lloyd mirror ----------

def test_surface_reflection_coefficient_is_minus_one():
    src = np.zeros(32000, dtype=np.float32)
    src[0] = 1.0
    _y, _H, meta = apply_three_path_channel(
        src, 32000, _geom(), np.random.default_rng(0)
    )
    assert meta["paths"]["surface"]["reflection_coefficient"] == -1.0


def test_lloyd_mirror_cancellation_at_dc():
    """Issue 4 deeper regression. At DC, the three-path channel response is
    real-valued and equals R_d * gain_d / r_d + R_s * gain_s / r_s + R_b * gain_b / r_b,
    where gain_i = 10^(-alpha(0) * r_i / 20000) and alpha(0) is the Thorpe
    DC floor (~3.3e-3 dB/km). With R_s = -1 and r_d ≈ r_s, the direct +
    surface contribution nearly cancels. Flipping R_s to +1 would make
    |H(0)| roughly 5x larger.
    """
    # Symmetric depths so r_d and r_s are very close.
    geo = _geom(source_depth_m=5.0, receiver_depth_m=5.0, horizontal_range_m=5_000.0)
    r_d, r_s, r_b = _slant_ranges(geo)
    R_b_linear = 10.0 ** (-geo.bottom_reflection_loss_db / 20.0)

    # Thorpe absorption at DC (non-zero — has a floor from the boric acid term).
    alpha_dc = float(thorpe_absorption_db_per_km(0.0))
    gain = lambda r: 10.0 ** (-alpha_dc * r / 20000.0)

    h_at_dc = float(np.real(three_path_response(0.0, geo)))

    expected_minus = (
        +1.0 * gain(r_d) / r_d
        + -1.0 * gain(r_s) / r_s
        + R_b_linear * gain(r_b) / r_b
    )
    expected_plus = (
        +1.0 * gain(r_d) / r_d
        + +1.0 * gain(r_s) / r_s
        + R_b_linear * gain(r_b) / r_b
    )

    assert abs(h_at_dc - expected_minus) < 1e-9, (
        f"module H(0)={h_at_dc:.6e} does not match analytical with surface=-1 "
        f"({expected_minus:.6e}); Lloyd-mirror sign is wrong"
    )
    # The surface=+1 alternative would be visibly different.
    assert abs(h_at_dc - expected_plus) > 1e-5


# ---------- Channel response basics ----------

def test_propagation_preserves_tonal_frequency():
    sr = 32_000
    t = np.arange(sr * 4) / sr
    src = np.sin(2.0 * np.pi * 100.0 * t).astype(np.float32)
    y, _H, _meta = apply_three_path_channel(src, sr, _geom(), np.random.default_rng(0))
    peak_bin = int(np.argmax(np.abs(np.fft.rfft(y))))
    peak_freq = peak_bin * sr / len(y)
    assert abs(peak_freq - 100.0) < 1.0


def test_spherical_spreading_dominates_long_range():
    """Without source-level boost, 50 km vs 5 km should differ by ~20 dB from
    spreading (1/r), with absorption adding only a few dB at 100 Hz."""
    sr = 32_000
    t = np.arange(sr * 4) / sr
    src = np.sin(2.0 * np.pi * 100.0 * t).astype(np.float32)
    y_near, *_ = apply_three_path_channel(src, sr, _geom(horizontal_range_m=5_000.0),  np.random.default_rng(0))
    y_far,  *_ = apply_three_path_channel(src, sr, _geom(horizontal_range_m=50_000.0), np.random.default_rng(0))
    rms_near = float(np.sqrt(np.mean(y_near.astype(np.float64) ** 2)))
    rms_far = float(np.sqrt(np.mean(y_far.astype(np.float64) ** 2)))
    ratio_db = 20.0 * np.log10(rms_near / rms_far)
    assert 15.0 < ratio_db < 25.0, f"spreading ratio outside band: {ratio_db:.2f} dB"


# ---------- Generator wiring: end-to-end ----------

def test_generate_clip_with_propagation_writes_geometry_to_manifest(synthetic_ambient_path, tmp_path):
    out = tmp_path / "c1_3_lite.wav"
    priors_t = TonalParameterPriors(n_sources_distribution={1: 1.0})
    r = generate_c1_1_clip(
        ambient_path=synthetic_ambient_path,
        out_path=out,
        seed=20260512,
        priors=priors_t,
        clip_duration_s=20.0,
        propagation_priors=PropagationGeometryPriors(),
    )
    manifest = r["manifest"]
    assert manifest.generator_version == C1_3_GENERATOR_VERSION
    assert manifest.lines, "expected at least one truth-manifest line with 1 forced source"
    line0 = manifest.lines[0]
    assert line0.propagation_model_id == PROPAGATION_MODEL_ID
    assert line0.propagation_geometry is not None
    assert line0.propagation_geometry.horizontal_range_m > 0


def test_generate_clip_propagation_audit_includes_c1_3_lite_deltas(synthetic_ambient_path, tmp_path):
    import json

    out = tmp_path / "c1_3_lite_audit.wav"
    priors_t = TonalParameterPriors(n_sources_distribution={1: 1.0})
    r = generate_c1_1_clip(
        ambient_path=synthetic_ambient_path,
        out_path=out,
        seed=20260512,
        priors=priors_t,
        clip_duration_s=20.0,
        propagation_priors=PropagationGeometryPriors(),
    )
    audit_path = Path(r["audit_path"])
    audit = json.loads(audit_path.read_text())
    snapshot = audit["parameter_snapshot"]
    assert snapshot["propagation_enabled"] is True
    assert snapshot["propagation"]["model_id"] == PROPAGATION_MODEL_ID
    delta_ids = {d["delta_id"] for d in snapshot["a1_3_3_deltas"]}
    expected = {d["delta_id"] for d in C1_3_LITE_DELTAS}
    assert expected.issubset(delta_ids), (
        f"audit deltas missing C1.3-lite entries: {expected - delta_ids}"
    )


def test_generate_clip_without_propagation_omits_geometry(synthetic_ambient_path, tmp_path):
    """Backwards compat: with propagation_priors=None the clip lands with no
    propagation_geometry on any line and the C1.1 / C1.2 generator versions."""
    out = tmp_path / "c1_1_baseline.wav"
    priors_t = TonalParameterPriors(n_sources_distribution={1: 1.0})
    r = generate_c1_1_clip(
        ambient_path=synthetic_ambient_path,
        out_path=out,
        seed=20260512,
        priors=priors_t,
        clip_duration_s=20.0,
    )
    assert r["manifest"].generator_version != C1_3_GENERATOR_VERSION
    for line in r["manifest"].lines:
        assert line.propagation_geometry is None
        assert line.propagation_model_id is None


# ---------- Issue 1: boost preserves received SNR at f0 ----------

def test_boost_preserves_received_snr_at_f0(synthetic_ambient_path, tmp_path):
    """Issue 1 regression. With boost-then-propagate, the mean h=0
    snr_curve_db with propagation should land within ~3 dB of the
    no-propagation baseline — the boost compensates for the channel
    loss at f0, so received SNR at the fundamental is preserved.
    """
    priors_t = TonalParameterPriors(n_sources_distribution={1: 1.0})
    common = dict(
        ambient_path=synthetic_ambient_path,
        seed=20260512,
        priors=priors_t,
        clip_duration_s=20.0,
    )

    r_no = generate_c1_1_clip(out_path=tmp_path / "no.wav", **common)
    r_pr = generate_c1_1_clip(
        out_path=tmp_path / "pr.wav",
        propagation_priors=PropagationGeometryPriors(),
        **common,
    )

    def _mean_h0_snr(r):
        lines = [l for l in r["manifest"].lines if l.harmonic_id == 0]
        if not lines or not lines[0].snr_curve_db:
            return None
        return float(np.mean(lines[0].snr_curve_db))

    snr_no = _mean_h0_snr(r_no)
    snr_pr = _mean_h0_snr(r_pr)
    assert snr_no is not None and snr_pr is not None, (
        "both clips must produce a non-empty h=0 SNR curve"
    )
    delta = abs(snr_pr - snr_no)
    assert delta < 3.0, (
        f"boost should preserve received SNR at f0: |snr_pr - snr_no| = {delta:.2f} dB "
        f"(snr_no={snr_no:.2f}, snr_pr={snr_pr:.2f})"
    )

    # And the boost identity: source_level_boost_db + channel_gain_at_f0_db ≈ 0
    prop_meta = r_pr["source_truths"][0]["propagation_metadata"]
    identity_residual = (
        prop_meta["source_level_boost_db"] + prop_meta["channel_gain_at_f0_db"]
    )
    assert abs(identity_residual) < 1e-6, (
        f"boost + channel_gain at f0 must sum to ~0; got {identity_residual:.6e}"
    )