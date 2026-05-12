"""Synthetic LOFAR generator.

B1 (`generate_b1_clip`): minimum-viable single-source deterministic injection.
C1.1 (`generate_c1_1_clip`): full A1 §3.3 parameterized multi-source generator
with decaying-cosine pulses, Rayleigh-jittered cluster timing, drift, and
weighted source-count sampling (including negative clips).

Both paths emit a triplet: WAV + truth manifest (A1 §3.3.1) + audit sidecar
with full provenance.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audit import make_provenance, write_audit_sidecar
from ..models import (
    StftConfig,
    SyntheticConfuserLabel,
    SyntheticLineGroundTruth,
    SyntheticPropagationGeometry,
    SyntheticTruthManifest,
)
from .ambient import load_deepship_ambient
from .biologicals import (
    BiologicalInjectionPriors,
    inject_biologicals,
    load_biological_library,
)
from .priors import (
    PropagationGeometryPriors,
    SampledPropagationGeometry,
    SampledTonalParameters,
    TonalParameterPriors,
    sample_n_sources,
    sample_propagation_geometry,
    sample_tonal_parameters,
)
from .propagation import (
    PROPAGATION_MODEL_ID,
    apply_three_path_channel,
    three_path_response,
)
from .tonals import inject_deterministic_tonal, inject_parameterized_tonal
from .truth import compute_per_frame_truth

LOG = logging.getLogger(__name__)

GENERATOR_VERSION = "0.1.0+b1"            # B1 path; do not change
C1_1_GENERATOR_VERSION = "0.2.0+c1.1"     # C1.1 path (no biologicals)
C1_2_GENERATOR_VERSION = "0.3.0+c1.2"     # C1.2 path (biologicals overlay enabled)
C1_3_GENERATOR_VERSION = "0.4.0+c1.3-lite"  # C1.3-lite path (propagation enabled)
TARGET_SR = 32000

A1_DELTAS = [
    {
        "delta_id": "f0_primary_floor_3hz",
        "rationale": "primary band [3, 500] Hz vs A1 [5, 500]; aligns with frozen Phase 1 baseline freq_min=3.0",
    },
    {
        "delta_id": "n_sources_distribution_weighted",
        "rationale": "categorical {0:0.15, 1:0.40, 2:0.30, 3:0.15} (A1 silent); includes negatives required for C2 binary classifier",
    },
    {
        "delta_id": "min_freq_separation_hz",
        "rationale": "rejection threshold 20 Hz (A1 silent); prevents physically unrealistic overlapping fundamentals",
    },
    {
        "delta_id": "pulses_per_cluster_range_invented",
        "rationale": "(1,5) inclusive uniform pulses per cluster (A1 silent); operational interpretation of A1 cluster timing",
    },
    {
        "delta_id": "source_id_schema_field",
        "rationale": "new optional source_id on SyntheticLineGroundTruth; enables Sprint 5 source-level (vs line-level) evaluation",
    },
]
C1_2_DELTAS = [
    {
        "delta_id": "biologicals_dclde_2018_only",
        "rationale": "DCLDE 2018 LF clips as initial biological source; Watkins / NOAA NRS / other libraries supported via the same BiologicalClipLibrary schema (no code change required)",
    },
    {
        "delta_id": "biological_overlay_priors_invented",
        "rationale": "n_biologicals categorical {0:0.40, 1:0.30, 2:0.20, 3:0.10}, per-overlay SNR uniform [3, 15] dB, cosine taper 0.3s at clip edges; A1 §3.2 silent on overlay parameters",
    },
    {
        "delta_id": "species_sqrt_weighting",
        "rationale": "default species sampling uses sqrt(library_count) per species — moderates extreme dataset imbalances (e.g., DCLDE 1089 Bm vs 16 Eg)",
    },
    {
        "delta_id": "confuser_clip_id_schema_rename",
        "rationale": "SyntheticConfuserLabel.watkins_id renamed to confuser_clip_id + source_dataset/species_code/target_snr_db fields added; original A1 schema implicitly Watkins-coupled per ENG feedback",
    },
]

C1_3_LITE_DELTAS = [
    {
        "delta_id": "parametric_three_path_substitutes_kraken_bellhop",
        "rationale": (
            "A1 §3.4 specified pre-computed KRAKEN/BELLHOP IR library "
            "(5 envs × 10 geometries × 2 bands = 100 IRs). C1.3-lite substitutes "
            "a parametric three-path (direct + surface + bottom) model with "
            "geometry sampled from priors. Team canonical-IR hunt 2026-05-12 "
            "confirmed no public 3-1000 Hz IR dataset exists — every measured-IR "
            "library targets underwater comms in the kHz range; SWellEx-96 is "
            "the sole public dataset where sim-to-real CIR validation is "
            "possible (Nannuru et al. SBL, IEEE JOE 2022), deferred to Sprint 5+."
        ),
    },
    {
        "delta_id": "isovelocity_sound_speed_baseline",
        "rationale": (
            "C1.3-lite uses single c=1500 m/s. A1 §3.4 (KRAKEN) would model "
            "SSP-dependent refraction; deferred to Sprint 5+ once "
            "canonical-environment IR set is computed."
        ),
    },
    {
        "delta_id": "thorpe_absorption_only",
        "rationale": (
            "Frequency-dependent volume absorption via Thorpe (1967); more "
            "comprehensive models (Francois-Garrison, Ainslie-McColm) deferred."
        ),
    },
    {
        "delta_id": "no_doppler",
        "rationale": (
            "Source/receiver radial velocities sub-bin at LOFAR resolution "
            "(narrowband Δf at 0.1 Hz bin width requires ≳10 kt closing for a "
            "100 Hz tonal); explicitly omitted from C1.3-lite scope."
        ),
    },
]

def _default_stft(sample_rate: int = TARGET_SR) -> StftConfig:
    """Default STFT for C1.1 truth-curve computation; matches build_synthetic_b1.py."""
    return StftConfig(
        sample_rate=sample_rate,
        n_fft=16384,
        hop_length=4096,
        window_length=16384,
        window="hanning",
    )


def generate_b1_clip(
    *,
    ambient_path: Path,
    out_path: Path,
    frequency_hz: float,
    t_start_s: float,
    t_end_s: float,
    target_snr_db: float,
    seed: int,
) -> dict:
    """Generate a single synthetic clip per B1 minimum-viable spec.

    Outputs alongside out_path:
    - <out_path>: combined synthetic audio (32 kHz mono WAV)
    - <out_path stem>.truth_manifest.json: A1 §3.3.1 ground-truth manifest
    - <out_path>.audit.json: provenance sidecar (Sprint 1 audit pattern)
    """
    ambient_waveform, source_sr = load_deepship_ambient(ambient_path, target_sr=TARGET_SR)

    combined, gt = inject_deterministic_tonal(
        ambient_waveform,
        sample_rate=TARGET_SR,
        frequency_hz=frequency_hz,
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        target_snr_db=target_snr_db,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), combined, samplerate=TARGET_SR, subtype="PCM_16")

    truth_lines = []
    for h_info in gt["harmonics"]:
        truth_lines.append(SyntheticLineGroundTruth(
            line_id=f"line_h{h_info['harmonic_id']}",
            source_type="tonal",
            harmonic_id=h_info["harmonic_id"],
            f0_hz=frequency_hz,  # fundamental, shared across harmonics
            freq_curve_hz=[h_info["harmonic_freq_hz"]],  # this harmonic's freq
            t_start_s=t_start_s,
            t_end_s=t_end_s,
            snr_curve_db=[h_info["snr_db"]],
            persistence_s=t_end_s - t_start_s,
            drift_rate_hz_per_s=0.0,
            mask_bin_indices=[],
            generation_seed=seed,
        ))
    manifest = SyntheticTruthManifest(
        clip_id=out_path.stem,
        lines=truth_lines,
        negative_label=False,
        confuser_labels=[],
        ambient_source_id=ambient_path.stem,
        ambient_source_clip_timestamp=None,
        propagation_environment_id=None,
        generator_version=GENERATOR_VERSION,
    )
    manifest_path = out_path.with_name(out_path.stem + ".truth_manifest.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    provenance = make_provenance(
        parameter_snapshot={
            "frequency_hz": frequency_hz,
            "t_start_s": t_start_s,
            "t_end_s": t_end_s,
            "target_snr_db": target_snr_db,
            "n_harmonics_injected": gt["n_harmonics_injected"],
            "harmonic_amplitude_decay": gt["harmonic_amplitude_decay"],
            "fade_s": gt["fade_s"],
            "local_ambient_rms_fundamental": gt["local_ambient_rms_fundamental"],
            "seed": seed,
            "generator_version": GENERATOR_VERSION,
            "source_sample_rate_hz": source_sr,
            "target_sample_rate_hz": TARGET_SR,
            "ambient_source_substitution": (
                "DeepShip vessel-free segment substituted for NOAA NRS per CEO "
                "direction 2026-05-10 (NRS acquisition pending; A1 §7 item 13 "
                "staged-implementation spirit preserved)."
            ),
            "operator_review_extensions_2026_05_10": (
                "B1 extended after operator review: local-bin SNR calculation "
                "(was global RMS), n_harmonics=3 with 0.7 decay (was 1 pure "
                "sinusoid), cosine fade gate (was hard step). All within "
                "minimum-viable scope; full A1 §3.3 parameterization remains C1."
            ),
        },
        source_recording_path=ambient_path,
    )
    audit_path = write_audit_sidecar(out_path, provenance)

    return {
        "wav_path": out_path,
        "manifest_path": manifest_path,
        "audit_path": audit_path,
        "ground_truth": gt,
    }


def generate_c1_1_clip(
    *,
    ambient_path: Path,
    out_path: Path,
    seed: int,
    priors: TonalParameterPriors | None = None,
    stft: StftConfig | None = None,
    clip_duration_s: float | None = None,
    biological_library_root: Path | None = None,
    biological_priors: BiologicalInjectionPriors | None = None,
    propagation_priors: PropagationGeometryPriors | None = None,
) -> dict:
    """Generate a single synthetic clip per C1.1 full A1 §3.3 parameterized spec.

    Pipeline:
      1. Load DeepShip ambient (NOAA NRS substitute per CEO 2026-05-10).
      2. Sample n_sources from priors.n_sources_distribution (may be 0 for negatives).
      3. For each source: rejection-sample params (min freq separation), inject
         decaying-cosine pulses via inject_parameterized_tonal. Each source's
         amplitude is computed against the ORIGINAL ambient (consistent SNR labels);
         source signals sum into the running combined.
      4. Compute per-frame truth (freq_curve, snr_curve, mask_bin_indices).
      5. Write WAV + truth manifest JSON + audit sidecar.

    Outputs alongside out_path:
      - <out_path>: combined synthetic audio (32 kHz mono WAV)
      - <out_path stem>.truth_manifest.json: A1 §3.3.1 ground-truth manifest
      - <out_path>.audit.json: provenance with priors + sampled-source snapshot + A1 deltas
    """
    priors = priors or TonalParameterPriors()
    stft = stft or _default_stft()
    rng = np.random.default_rng(seed)

    ambient_waveform, source_sr = load_deepship_ambient(ambient_path, target_sr=TARGET_SR)

    if clip_duration_s is not None:
        max_samples = int(clip_duration_s * TARGET_SR)
        if len(ambient_waveform) > max_samples:
            ambient_waveform = ambient_waveform[:max_samples]
    clip_duration_s = len(ambient_waveform) / TARGET_SR

    if len(ambient_waveform) < stft.window_length:
        raise ValueError(
            f"ambient {len(ambient_waveform)} samples < stft.window_length {stft.window_length}; "
            "cannot compute per-frame truth"
        )

    n_sources_sampled = sample_n_sources(rng, priors)

    source_truths: list[dict] = []
    source_ids: list[str] = []
    sampled_params_list: list[SampledTonalParameters] = []
    drawn_f0s: list[float] = []

    running_combined = ambient_waveform.copy()

    # source_id -> propagation geometry + path metadata, populated only when
    # propagation_priors is not None.
    propagation_geometries: dict[str, SyntheticPropagationGeometry] = {}
    propagation_meta_by_source: dict[str, dict] = {}

    for i in range(n_sources_sampled):
        params = sample_tonal_parameters(
            rng, priors, clip_duration_s, prior_f0s_hz=tuple(drawn_f0s),
        )
        if params is None:
            LOG.warning(
                "C1.1: source %d/%d failed rejection sampling (drawn_f0s=%s); skipping",
                i, n_sources_sampled, drawn_f0s,
            )
            continue

        source_id = f"src_{i:02d}"
        try:
            this_combined, source_truth = inject_parameterized_tonal(
                ambient_waveform, TARGET_SR, params=params, rng=rng,
            )
        except ValueError as e:
            LOG.warning(
                "C1.1: source %d (f0=%.1f Hz) injection failed: %s; skipping",
                i, params.f0_hz, e,
            )
            continue

        # Isolate the injected source signal so propagation operates on the
        # source alone (not the ambient + source mix).
        source_signal = this_combined - ambient_waveform

        if propagation_priors is not None:
            geometry = sample_propagation_geometry(rng, propagation_priors)
            if geometry is None:
                LOG.warning(
                    "C1.3-lite: geometry rejection exhausted for source %d; "
                    "falling back to no-propagation for this source",
                    i,
                )
            else:
                # Boost source level to compensate for channel loss at f0, so
                # `target_snr_db` keeps its C1.1 semantic ("desired received
                # SNR"). After propagation the fundamental lands ≈ target;
                # harmonics scatter around it via per-frequency channel
                # differential — physically motivated multipath modulation.
                # (team review 2026-05-12 Issue 1 — preferred resolution.)
                channel_gain_at_f0 = float(np.abs(
                    three_path_response(np.array([params.f0_hz]), geometry)[0]
                ))
                source_level_boost = 1.0 / max(channel_gain_at_f0, 1e-12)

                source_signal = source_signal * source_level_boost
                source_signal, _H_rfft, prop_meta = apply_three_path_channel(
                    source_signal, TARGET_SR, geometry, rng,
                )

                # Update stored amplitudes to reflect post-boost-and-propagation
                # received levels. truth.compute_per_frame_truth reads
                # source_truth["harmonics"][h]["amplitude"] (after Edit 1).
                harmonic_freqs = np.array(
                    [hd["harmonic_freq_hz"] for hd in source_truth["harmonics"]],
                    dtype=np.float64,
                )
                gains = np.abs(three_path_response(harmonic_freqs, geometry))
                net_gain = source_level_boost * gains  # net[0] ≈ 1.0 at f0

                source_truth["target_source_snr_db"] = source_truth["target_snr_db"]
                source_truth["fundamental_amplitude"] *= float(net_gain[0])
                for hd, factor in zip(source_truth["harmonics"], net_gain):
                    hd["amplitude"] *= float(factor)
                    hd["received_snr_db"] = float(
                        hd["snr_db"] + 20.0 * np.log10(max(factor, 1e-12))
                    )

                # Boost + per-f0 gain become part of the audit trail so any
                # downstream observer can reconstruct exactly what was applied.
                prop_meta["source_level_boost_db"] = float(
                    20.0 * np.log10(source_level_boost)
                )
                prop_meta["channel_gain_at_f0_db"] = float(
                    20.0 * np.log10(max(channel_gain_at_f0, 1e-12))
                )

                source_truth["propagation_model_id"] = PROPAGATION_MODEL_ID
                source_truth["propagation_geometry"] = asdict(geometry)
                source_truth["propagation_metadata"] = prop_meta

                propagation_geometries[source_id] = SyntheticPropagationGeometry(
                    **asdict(geometry)
                )
                propagation_meta_by_source[source_id] = prop_meta

        running_combined = running_combined + source_signal

        source_truths.append(source_truth)
        source_ids.append(source_id)
        sampled_params_list.append(params)
        drawn_f0s.append(params.f0_hz)

    n_sources_realized = len(source_truths)
    is_negative = (n_sources_realized == 0)

    if is_negative:
        truth_lines: list[SyntheticLineGroundTruth] = []
    else:
        truth_lines = compute_per_frame_truth(
            source_truths=source_truths,
            source_ids=source_ids,
            ambient=ambient_waveform,
            stft=stft,
            generation_seed=seed,
        )
        # Stamp propagation geometry onto every line whose source went through
        # the three-path channel. compute_per_frame_truth itself stays
        # propagation-agnostic; this is the integration boundary.
        if propagation_geometries:
            truth_lines = [
                line.model_copy(update={
                    "propagation_geometry": propagation_geometries[line.source_id],
                    "propagation_model_id": PROPAGATION_MODEL_ID,
                })
                if line.source_id in propagation_geometries
                else line
                for line in truth_lines
            ]

    # ---- C1.2: biological confuser overlay (optional) ----
    confuser_labels: list[SyntheticConfuserLabel] = []
    biological_overlays_metadata: list[dict] = []
    biological_library_id: str | None = None
    biological_priors_resolved: BiologicalInjectionPriors | None = None

    if biological_library_root is not None:
        biological_priors_resolved = biological_priors or BiologicalInjectionPriors()
        library = load_biological_library(Path(biological_library_root))
        biological_library_id = library.library_id
        running_combined, bio_overlays = inject_biologicals(
            running_combined,
            TARGET_SR,
            library=library,
            library_root=Path(biological_library_root),
            priors=biological_priors_resolved,
            rng=rng,
        )
        for ov in bio_overlays:
            confuser_labels.append(SyntheticConfuserLabel(
                species=ov.species_name,
                species_code=ov.species_code,
                confuser_clip_id=ov.clip_id,
                source_dataset=ov.source_dataset,
                t_start_s=ov.t_onset_s,
                t_end_s=ov.t_onset_s + ov.duration_s,
                freq_range_hz=ov.freq_range_hz,
                target_snr_db=ov.target_snr_db,
            ))
            biological_overlays_metadata.append({
                "clip_id": ov.clip_id,
                "species_code": ov.species_code,
                "source_dataset": ov.source_dataset,
                "t_onset_s": ov.t_onset_s,
                "duration_s": ov.duration_s,
                "target_snr_db": ov.target_snr_db,
                "freq_range_hz": list(ov.freq_range_hz),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), running_combined, samplerate=TARGET_SR, subtype="PCM_16")

    manifest_version = (
        C1_3_GENERATOR_VERSION if propagation_priors is not None
        else C1_2_GENERATOR_VERSION if biological_library_root is not None
        else C1_1_GENERATOR_VERSION
    )
    manifest = SyntheticTruthManifest(
        clip_id=out_path.stem,
        lines=truth_lines,
        negative_label=is_negative,
        confuser_labels=confuser_labels,
        ambient_source_id=ambient_path.stem,
        ambient_source_clip_timestamp=None,
        propagation_environment_id=None,
        generator_version=manifest_version,
    )
    manifest_path = out_path.with_name(out_path.stem + ".truth_manifest.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    priors_snapshot = asdict(priors)
    priors_snapshot["n_sources_distribution"] = {
        str(k): float(v) for k, v in priors.n_sources_distribution.items()
    }
    sampled_params_snapshot = [
        {"source_id": sid, **asdict(p)}
        for sid, p in zip(source_ids, sampled_params_list)
    ]

    bio_priors_snapshot: dict | None = None
    if biological_priors_resolved is not None:
        bio_priors_snapshot = asdict(biological_priors_resolved)
        bio_priors_snapshot["n_biologicals_distribution"] = {
            str(k): float(v)
            for k, v in biological_priors_resolved.n_biologicals_distribution.items()
        }

    deltas = list(A1_DELTAS)
    if biological_library_root is not None:
        deltas = deltas + C1_2_DELTAS
    if propagation_priors is not None:
        deltas = deltas + C1_3_LITE_DELTAS

    propagation_snapshot: dict | None = None
    if propagation_priors is not None:
        propagation_snapshot = {
            "model_id": PROPAGATION_MODEL_ID,
            "priors": asdict(propagation_priors),
            "geometries_by_source": {
                sid: g.model_dump() for sid, g in propagation_geometries.items()
            },
            "path_metadata_by_source": propagation_meta_by_source,
        }

    provenance = make_provenance(
        parameter_snapshot={
            "seed": seed,
            "generator_version": manifest_version,
            "source_sample_rate_hz": source_sr,
            "target_sample_rate_hz": TARGET_SR,
            "clip_duration_s": clip_duration_s,
            "n_sources_sampled": n_sources_sampled,
            "n_sources_realized": n_sources_realized,
            "negative_label": is_negative,
            "priors": priors_snapshot,
            "sampled_sources": sampled_params_snapshot,
            "stft": stft.model_dump(),
            "biologicals_enabled": biological_library_root is not None,
            "biological_library_id": biological_library_id,
            "biological_library_root": (
                str(biological_library_root) if biological_library_root is not None else None
            ),
            "biological_priors": bio_priors_snapshot,
            "n_biologicals_realized": len(confuser_labels),
            "biological_overlays": biological_overlays_metadata,
            "propagation_enabled": propagation_priors is not None,
            "propagation": propagation_snapshot,
            "a1_3_3_deltas": deltas,
            "ambient_source_substitution": (
                "DeepShip vessel-free segment substituted for NOAA NRS per CEO "
                "direction 2026-05-10 (NRS acquisition pending; A1 §7 item 13 "
                "staged-implementation spirit preserved)."
            ),
        },
        source_recording_path=ambient_path,
    )
    audit_path = write_audit_sidecar(out_path, provenance)

    return {
        "wav_path": out_path,
        "manifest_path": manifest_path,
        "audit_path": audit_path,
        "n_sources_sampled": n_sources_sampled,
        "n_sources_realized": n_sources_realized,
        "negative_label": is_negative,
        "n_biologicals_realized": len(confuser_labels),
        "biologicals_enabled": biological_library_root is not None,
        "source_truths": source_truths,
        "manifest": manifest,
    }