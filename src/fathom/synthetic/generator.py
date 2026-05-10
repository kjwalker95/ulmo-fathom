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

    line = SyntheticLineGroundTruth(
        line_id="line_0",
        source_type="tonal",
        harmonic_id=0,
        f0_hz=frequency_hz,
        freq_curve_hz=[frequency_hz],
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        snr_curve_db=[target_snr_db],
        persistence_s=t_end_s - t_start_s,
        drift_rate_hz_per_s=0.0,
        mask_bin_indices=[],  # B1 leaves empty; C1 computes precise mask
        generation_seed=seed,
    )
    manifest = SyntheticTruthManifest(
        clip_id=out_path.stem,
        lines=[line],
        negative_label=False,
        confuser_labels=[],
        ambient_source_id=ambient_path.stem,
        ambient_source_clip_timestamp=None,
        propagation_environment_id=None,  # B1 has no propagation
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
            "seed": seed,
            "generator_version": GENERATOR_VERSION,
            "source_sample_rate_hz": source_sr,
            "target_sample_rate_hz": TARGET_SR,
            "ambient_source_substitution": (
                "DeepShip vessel-free segment substituted for NOAA NRS per CEO direction "
                "2026-05-10 (NRS acquisition pending; A1 §7 item 13 staged-implementation "
                "spirit preserved)."
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