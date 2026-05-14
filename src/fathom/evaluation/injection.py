"""Tier-2 real-ambient injection: synthetic tonals into real DeepShip ambient.

Sprint 5 Cluster C1 (A3 §3.1 Tier-2). Per the design memo:
- Real ambient already contains real propagation effects, so we do NOT apply
  the C1.3-lite three-path channel on injected tonals. Double-propagation
  would produce unrealistic multipath interference.
- Output triplet (WAV + truth_manifest.json + wav.audit.json) matches the
  C1.1 synthetic-clip schema, so SyntheticPatchDataset + ml_eval.evaluate_model
  run unmodified on Tier-2 evaluation data.
- Inject at calibrated received SNR via inject_parameterized_tonal's default
  boost-then-no-propagate path.

Two modes:
  - Sampled: pass tonal_priors (+ optional n_sources override). Priors drive
    per-source params via sample_tonal_parameters.
  - Explicit: pass explicit_params: list[SampledTonalParameters] to bypass
    priors entirely (used by the smoke test for SNR control).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from fathom.audit import make_provenance, write_audit_sidecar
from fathom.detection.ml_data import default_lofar_config
from fathom.models import (
    StftConfig,
    SyntheticLineGroundTruth,
    SyntheticTruthManifest,
)
from fathom.synthetic.ambient import load_deepship_ambient
from fathom.synthetic.priors import (
    SampledTonalParameters,
    TonalParameterPriors,
    sample_n_sources,
    sample_tonal_parameters,
)
from fathom.synthetic.tonals import inject_parameterized_tonal
from fathom.synthetic.truth import compute_per_frame_truth

TIER2_GENERATOR_VERSION = "tier2_real_ambient_injection_v1"


@dataclass(frozen=True)
class Tier2InjectionResult:
    """Output of one Tier-2 injection: paths + truth manifest in memory."""
    wav_path: Path
    manifest_path: Path
    audit_path: Path
    manifest: SyntheticTruthManifest
    n_sources_realized: int


def inject_into_real_ambient(
    ambient_path: Path,
    *,
    out_dir: Path,
    clip_id: str,
    seed: int,
    tonal_priors: TonalParameterPriors | None = None,
    explicit_params: list[SampledTonalParameters] | None = None,
    n_sources: int | None = None,
    clip_duration_s: float | None = None,
    target_sample_rate: int = 32_000,
    stft: StftConfig | None = None,
) -> Tier2InjectionResult:
    """Inject synthetic tonals into real DeepShip ambient (no propagation).

    Modes are mutually exclusive: pass exactly one of `tonal_priors` or
    `explicit_params`.

    Returns a Tier2InjectionResult; writes three files under out_dir:
      - <clip_id>.wav                   (combined synthetic+ambient audio)
      - <clip_id>.truth_manifest.json   (A1 §3.3.1 SyntheticTruthManifest)
      - <clip_id>.wav.audit.json        (provenance + priors + sampled params)
    """
    if (tonal_priors is None) == (explicit_params is None):
        raise ValueError(
            "exactly one of tonal_priors / explicit_params must be provided"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    if stft is None:
        stft = default_lofar_config(sample_rate=target_sample_rate).stft

    ambient, _source_sr = load_deepship_ambient(
        ambient_path, target_sr=target_sample_rate
    )

    if clip_duration_s is not None:
        max_samples = int(clip_duration_s * target_sample_rate)
        if len(ambient) > max_samples:
            ambient = ambient[:max_samples]
    actual_clip_duration_s = len(ambient) / target_sample_rate

    if len(ambient) < stft.window_length:
        raise ValueError(
            f"ambient {len(ambient)} samples < stft.window_length "
            f"{stft.window_length}; cannot compute per-frame truth"
        )

    # Resolve source params (sampled vs explicit).
    if explicit_params is not None:
        params_list = list(explicit_params)
    else:
        n = (
            sample_n_sources(rng, tonal_priors)
            if n_sources is None else n_sources
        )
        params_list = []
        drawn_f0s: list[float] = []
        for _ in range(n):
            p = sample_tonal_parameters(
                rng, tonal_priors, actual_clip_duration_s,
                prior_f0s_hz=tuple(drawn_f0s),
            )
            if p is not None:
                params_list.append(p)
                drawn_f0s.append(p.f0_hz)

    # Per-source injection. No propagation (Sprint 5 §C1 design).
    source_truths: list[dict] = []
    source_ids: list[str] = []
    running_combined = ambient.copy()
    sampled_params_snapshot: list[dict] = []

    for i, params in enumerate(params_list):
        source_id = f"src_{i:02d}"
        try:
            this_combined, source_truth = inject_parameterized_tonal(
                ambient, target_sample_rate, params=params, rng=rng,
            )
        except ValueError:
            continue
        running_combined = running_combined + (this_combined - ambient)
        source_truths.append(source_truth)
        source_ids.append(source_id)
        sampled_params_snapshot.append({
            "source_id": source_id,
            "f0_hz": params.f0_hz,
            "n_harmonics": params.n_harmonics,
            "target_snr_db": params.target_snr_db,
            "drift_rate_hz_per_s": params.drift_rate_hz_per_s,
            "total_persistence_s": params.total_persistence_s,
            "t_onset_s": params.t_onset_s,
        })

    n_realized = len(source_truths)
    is_negative = n_realized == 0

    gt_rows: list[SyntheticLineGroundTruth] = []
    if n_realized > 0:
        gt_rows = compute_per_frame_truth(
            source_truths=source_truths,
            source_ids=source_ids,
            ambient=ambient,
            stft=stft,
            generation_seed=seed,
        )

    wav_path = out_dir / f"{clip_id}.wav"
    sf.write(
        str(wav_path),
        running_combined.astype(np.float32),
        target_sample_rate,
        subtype="PCM_16",
    )

    manifest = SyntheticTruthManifest(
        clip_id=clip_id,
        lines=gt_rows,
        negative_label=is_negative,
        confuser_labels=[],
        ambient_source_id=str(ambient_path),
        ambient_source_clip_timestamp=None,
        propagation_environment_id=None,
        generator_version=TIER2_GENERATOR_VERSION,
    )
    manifest_path = wav_path.with_name(wav_path.stem + ".truth_manifest.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    provenance = make_provenance(
        parameter_snapshot={
            "tier2_injection_v1": True,
            "ambient_path": str(ambient_path),
            "target_sample_rate": target_sample_rate,
            "clip_duration_s": actual_clip_duration_s,
            "seed": seed,
            "source_mode": (
                "explicit" if explicit_params is not None else "sampled"
            ),
            "n_sources_realized": n_realized,
            "sampled_params": sampled_params_snapshot,
            "tonal_priors_snapshot": (
                asdict(tonal_priors) if tonal_priors is not None else None
            ),
            "propagation_applied": False,
            "propagation_rationale": (
                "Real ambient already propagated; Sprint 5 §C1 design "
                "avoids double-propagation."
            ),
        },
        source_recording_path=ambient_path,
    )
    audit_path = write_audit_sidecar(wav_path, provenance)

    return Tier2InjectionResult(
        wav_path=wav_path,
        manifest_path=manifest_path,
        audit_path=audit_path,
        manifest=manifest,
        n_sources_realized=n_realized,
    )
