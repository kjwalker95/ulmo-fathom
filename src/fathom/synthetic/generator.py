"""B1 minimum-viable synthetic LOFAR generator.

Loads ambient (DeepShip stand-in for NOAA NRS pending acquisition) → injects
a deterministic tonal → writes WAV + truth manifest (A1 §3.3.1) + audit
sidecar. C1 expands to the full layered model (biologicals + KRAKEN/BELLHOP
IRs + drift + harmonic structure).
"""
from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf

from ..audit import make_provenance, write_audit_sidecar
from ..models import (
    SyntheticLineGroundTruth,
    SyntheticTruthManifest,
)
from .ambient import load_deepship_ambient
from .tonals import inject_deterministic_tonal

LOG = logging.getLogger(__name__)

GENERATOR_VERSION = "0.1.0+b1"
TARGET_SR = 32000


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