"""Tier-2 evaluation dataset orchestrator (Sprint 5 Cluster C1).

Walks a single split partition (val/test) of the post-A0 SplitManifest,
optionally narrowed to a vessel subset, and produces a Tier-2 evaluation
dataset by calling inject_into_real_ambient n_clips_per_vessel times per
vessel with seeded variation.

Output structure mirrors training_dataset_v2's:
  <out_dir>/
    manifest.json              -- top-level (vessels, clips, priors)
    <clip_id>.wav
    <clip_id>.truth_manifest.json
    <clip_id>.wav.audit.json

ml_eval.evaluate_model consumes this directly via SyntheticPatchDataset
because the per-clip schema matches A1 §3.3.1.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fathom.evaluation.injection import (
    Tier2InjectionResult,
    inject_into_real_ambient,
)
from fathom.models import SplitManifest
from fathom.synthetic.priors import TonalParameterPriors


@dataclass(frozen=True)
class Tier2DatasetManifest:
    """Top-level manifest for a Tier-2 evaluation dataset."""
    dataset_id: str
    split_manifest_path: str
    partition: str
    vessel_subset_path: str | None
    n_vessels: int
    n_clips_per_vessel: int
    n_clips_total: int
    seed: int
    target_sample_rate: int
    clip_duration_s: float | None
    tonal_priors: dict
    clips: list[dict]
    built_at: str


def _load_partition_compound_keys(
    manifest_path: Path, partition: str
) -> list[str]:
    raw = json.loads(manifest_path.read_text())
    manifest = SplitManifest.model_validate(raw)
    field_map = {
        "train": manifest.train_vessels,
        "val": manifest.val_vessels,
        "test": manifest.test_vessels,
    }
    if partition not in field_map:
        raise ValueError(
            f"partition must be one of {sorted(field_map)}; got {partition!r}"
        )
    keys = list(field_map[partition])
    bare = [k for k in keys if "/" not in k]
    if bare:
        raise ValueError(
            f"manifest at {manifest_path} contains bare-ID keys "
            f"{sorted(bare)[:5]}...; regenerate with Sprint 5 A0 build_splits.py"
        )
    return keys


def _resolve_compound_key_to_wav(
    compound_key: str, deepship_root: Path
) -> Path:
    """`Cargo/103` -> <deepship_root>/Cargo/103.wav. Raises if missing."""
    class_label, stem = compound_key.split("/", 1)
    wav = deepship_root / class_label / f"{stem}.wav"
    if not wav.exists():
        raise FileNotFoundError(
            f"compound key {compound_key!r} did not resolve to a file: {wav}"
        )
    return wav


def build_tier2_dataset(
    split_manifest_path: Path,
    partition: Literal["val", "test"],
    *,
    deepship_root: Path,
    n_clips_per_vessel: int,
    seed: int,
    tonal_priors: TonalParameterPriors,
    out_dir: Path,
    vessel_subset_path: Path | None = None,
    clip_duration_s: float | None = None,
    target_sample_rate: int = 32_000,
) -> Tier2DatasetManifest:
    """Build a Tier-2 evaluation dataset from a SplitManifest partition.

    Per Sprint5_Plan §C1: real ambient is drawn from the requested partition;
    synthetic tonals are injected per priors (no propagation); per-clip schema
    matches A1 §3.3.1 so existing eval code runs unmodified.

    If `vessel_subset_path` is provided, only the compound keys in that JSON
    file's `c4_c6_subset` array are used — Sprint5_Plan §C4 mechanism to limit
    the test-vessel touch to the 6-vessel C4/C6 subset (the other 6 stay
    reserved for Sprint 6 Tier-3 operator labeling).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    partition_keys = _load_partition_compound_keys(split_manifest_path, partition)

    if vessel_subset_path is not None:
        subset_payload = json.loads(Path(vessel_subset_path).read_text())
        subset_keys = set(subset_payload.get("c4_c6_subset", []))
        if not subset_keys:
            raise ValueError(
                f"vessel_subset_path {vessel_subset_path} missing or empty "
                "'c4_c6_subset' field"
            )
        unmatched = subset_keys - set(partition_keys)
        if unmatched:
            raise ValueError(
                f"vessel_subset keys not in {partition} partition: "
                f"{sorted(unmatched)}"
            )
        partition_keys = [k for k in partition_keys if k in subset_keys]

    if not partition_keys:
        raise ValueError(
            f"no vessels in {partition} partition (after subset filter, if any)"
        )

    clips_meta: list[dict] = []
    for vessel_idx, compound_key in enumerate(partition_keys):
        wav = _resolve_compound_key_to_wav(compound_key, deepship_root)
        for clip_idx in range(n_clips_per_vessel):
            clip_seed = seed + vessel_idx * 1_000 + clip_idx
            class_label, stem = compound_key.split("/", 1)
            clip_id = f"tier2_{class_label}_{stem}_c{clip_idx:02d}"

            result: Tier2InjectionResult = inject_into_real_ambient(
                ambient_path=wav,
                out_dir=out_dir,
                clip_id=clip_id,
                seed=clip_seed,
                tonal_priors=tonal_priors,
                clip_duration_s=clip_duration_s,
                target_sample_rate=target_sample_rate,
            )
            clips_meta.append({
                "clip_id": clip_id,
                "vessel_compound_key": compound_key,
                "wav_relative_path": result.wav_path.relative_to(out_dir).as_posix(),
                "manifest_relative_path": (
                    result.manifest_path.relative_to(out_dir).as_posix()
                ),
                "audit_relative_path": (
                    result.audit_path.relative_to(out_dir).as_posix()
                ),
                "n_sources_realized": result.n_sources_realized,
                "n_lines": len(result.manifest.lines),
                "seed": clip_seed,
            })

    dataset_manifest = Tier2DatasetManifest(
        dataset_id=out_dir.name,
        split_manifest_path=str(split_manifest_path),
        partition=partition,
        vessel_subset_path=str(vessel_subset_path) if vessel_subset_path else None,
        n_vessels=len(partition_keys),
        n_clips_per_vessel=n_clips_per_vessel,
        n_clips_total=len(clips_meta),
        seed=seed,
        target_sample_rate=target_sample_rate,
        clip_duration_s=clip_duration_s,
        tonal_priors=asdict(tonal_priors),
        clips=clips_meta,
        built_at=datetime.now(tz=timezone.utc).isoformat(),
    )

    (out_dir / "manifest.json").write_text(
        json.dumps(asdict(dataset_manifest), indent=2)
    )

    return dataset_manifest