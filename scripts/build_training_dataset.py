"""Bulk synthetic dataset generator for ML detector training.

Loops generate_c1_1_clip across N clips with configurable A1 §3.3 priors
+ DCLDE biological confuser overlay. No per-clip PNGs — training doesn't
need them.

Writes per-clip WAV + truth manifest + audit sidecar into out_dir, plus a
top-level manifest.json listing all clips for reproducibility (Sprint 3
SplitManifest pattern).

Sprint 5 additions (2026-05-13):
- Prior overrides via CLI flags (drift, harmonics, harmonic decay,
  persistence). Defaults match v1 / TonalParameterPriors defaults.
- Vessel-partition discipline: --split-manifest + --split-partition limit
  ambient sourcing to a single train/val/test partition of compound keys
  (post-A0 SplitManifest format). Required for v2 generation; absent
  filter preserves v1 behavior (sample from all recordings).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from fathom.models import SplitManifest
from fathom.synthetic import generate_c1_1_clip
from fathom.synthetic.priors import TonalParameterPriors

CONSOLE = Console()


def _load_partition_compound_keys(
    manifest_path: Path, partition: str
) -> set[str]:
    """Load a SplitManifest and return compound keys for a single partition.

    Compound keys are `<class>/<vessel_id>` per Sprint 5 A0. Caller filters
    candidate ambient WAV paths by `(parent.name, path.stem)` membership.
    """
    raw = json.loads(manifest_path.read_text())
    manifest = SplitManifest.model_validate(raw)
    field_map = {
        "train": manifest.train_vessels,
        "val": manifest.val_vessels,
        "test": manifest.test_vessels,
    }
    if partition not in field_map:
        raise click.BadParameter(
            f"--split-partition must be one of {sorted(field_map)}; got {partition!r}"
        )
    keys = set(field_map[partition])
    bare = [k for k in keys if "/" not in k]
    if bare:
        raise click.UsageError(
            f"manifest at {manifest_path} contains bare-ID keys "
            f"{sorted(bare)[:5]}...; regenerate with Sprint 5 A0 build_splits.py "
            "to produce compound `<class>/<vessel_id>` keys"
        )
    return keys


def _list_ambient_wavs(
    ambient_dir: Path, partition_keys: set[str] | None
) -> list[Path]:
    """Recursively enumerate ambient WAVs, optionally filtered to a split partition.

    When partition_keys is None, returns all WAVs (v1 behavior). When provided,
    keeps only WAVs whose `(parent.name, path.stem)` compound key is in the set.
    """
    all_wavs = sorted(ambient_dir.rglob("*.wav"))
    if partition_keys is None:
        return all_wavs
    matched: list[Path] = []
    for wav in all_wavs:
        compound = f"{wav.parent.name}/{wav.stem}"
        if compound in partition_keys:
            matched.append(wav)
    return matched


def _build_priors(
    drift_rate_std_hz_per_s: float,
    n_harmonics_max: int,
    harmonic_decay_min: float,
    harmonic_decay_max: float,
    persistence_log_min: float,
    persistence_log_max: float,
) -> TonalParameterPriors:
    """Construct TonalParameterPriors from CLI overrides.

    Defaults at the click-option layer match TonalParameterPriors dataclass
    defaults, so a v1-style invocation (no override flags) produces v1-equivalent
    priors.
    """
    if n_harmonics_max < 1:
        raise click.BadParameter(f"--n-harmonics-max must be >= 1; got {n_harmonics_max}")
    if not (0.0 < harmonic_decay_min <= harmonic_decay_max <= 1.0):
        raise click.BadParameter(
            f"harmonic decay range must satisfy 0 < min <= max <= 1; "
            f"got ({harmonic_decay_min}, {harmonic_decay_max})"
        )
    if not (persistence_log_min < persistence_log_max):
        raise click.BadParameter(
            f"persistence-log range must satisfy min < max; "
            f"got ({persistence_log_min}, {persistence_log_max})"
        )
    return TonalParameterPriors(
        n_harmonics_choices=tuple(range(1, n_harmonics_max + 1)),
        harmonic_decay_range=(harmonic_decay_min, harmonic_decay_max),
        total_persistence_log_range=(persistence_log_min, persistence_log_max),
        drift_rate_std_hz_per_s=drift_rate_std_hz_per_s,
    )


@click.command()
@click.option(
    "--ambient-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory of ambient WAV recordings (recursively scanned).",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("data/training_dataset_v1"),
)
@click.option("--n-clips", type=int, default=500)
@click.option("--seed", type=int, default=20260512)
@click.option(
    "--biological-library-root",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="DCLDE BiologicalClipLibrary root.",
)
@click.option(
    "--split-manifest",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional SplitManifest JSON (post-A0 compound-key format). When set, "
    "--split-partition is required and ambient sourcing is limited to "
    "vessels in that partition. Omit to preserve v1 unfiltered behavior.",
)
@click.option(
    "--split-partition",
    type=click.Choice(["train", "val", "test"]),
    default=None,
    help="Which partition of the SplitManifest to draw ambient from. Required "
    "when --split-manifest is set.",
)
@click.option(
    "--drift-rate-std-hz-per-s",
    type=float,
    default=0.05,
    help="TonalParameterPriors.drift_rate_std_hz_per_s. v1=0.05; v2=1.0.",
)
@click.option(
    "--n-harmonics-max",
    type=int,
    default=3,
    help="Max number of harmonics; choices become tuple(range(1, max+1)). "
    "v1=3; v2=6.",
)
@click.option(
    "--harmonic-decay-min",
    type=float,
    default=0.3,
    help="Lower bound of TonalParameterPriors.harmonic_decay_range. v1=0.3; v2=0.2.",
)
@click.option(
    "--harmonic-decay-max",
    type=float,
    default=0.7,
    help="Upper bound of TonalParameterPriors.harmonic_decay_range. v1=0.7; v2=0.8.",
)
@click.option(
    "--persistence-log-min",
    type=float,
    default=1.0,
    help="Lower bound of TonalParameterPriors.total_persistence_log_range. "
    "v1=1.0; v2=10.0.",
)
@click.option(
    "--persistence-log-max",
    type=float,
    default=120.0,
    help="Upper bound of TonalParameterPriors.total_persistence_log_range. "
    "v1=120.0; v2=180.0.",
)
def main(
    ambient_dir: Path,
    out_dir: Path,
    n_clips: int,
    seed: int,
    biological_library_root: Path,
    split_manifest: Path | None,
    split_partition: str | None,
    drift_rate_std_hz_per_s: float,
    n_harmonics_max: int,
    harmonic_decay_min: float,
    harmonic_decay_max: float,
    persistence_log_min: float,
    persistence_log_max: float,
) -> None:
    """Generate a bulk synthetic training dataset (configurable A1 priors + bios)."""
    if (split_manifest is None) != (split_partition is None):
        raise click.UsageError(
            "--split-manifest and --split-partition must be used together"
        )

    partition_keys: set[str] | None = None
    if split_manifest is not None:
        partition_keys = _load_partition_compound_keys(split_manifest, split_partition)
        CONSOLE.print(
            f"[cyan]Split filter: {split_partition} partition of "
            f"{split_manifest} -> {len(partition_keys)} vessels[/cyan]"
        )

    priors = _build_priors(
        drift_rate_std_hz_per_s=drift_rate_std_hz_per_s,
        n_harmonics_max=n_harmonics_max,
        harmonic_decay_min=harmonic_decay_min,
        harmonic_decay_max=harmonic_decay_max,
        persistence_log_min=persistence_log_min,
        persistence_log_max=persistence_log_max,
    )
    CONSOLE.print(
        f"[cyan]Priors: drift_std={priors.drift_rate_std_hz_per_s} "
        f"n_harmonics={priors.n_harmonics_choices} "
        f"decay={priors.harmonic_decay_range} "
        f"persistence_log={priors.total_persistence_log_range}[/cyan]"
    )

    ambient_paths = _list_ambient_wavs(ambient_dir, partition_keys)
    if not ambient_paths:
        raise click.UsageError(
            f"no .wav files matched under {ambient_dir}"
            f"{' (after partition filter)' if partition_keys else ''}"
        )
    CONSOLE.print(
        f"[cyan]Ambient sources: {len(ambient_paths)} files under {ambient_dir}"
        f"{' (partition-filtered)' if partition_keys else ''}[/cyan]"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    chooser_rng = np.random.default_rng(seed)
    chosen_indices = chooser_rng.choice(len(ambient_paths), size=n_clips, replace=True)

    clip_summaries: list[dict] = []
    summary_stats = {
        "negatives": 0,
        "positives_by_n_sources": {1: 0, 2: 0, 3: 0},
        "total_lines": 0,
        "total_biologicals": 0,
        "snr_buckets": {"<5": 0, "5-10": 0, "10-15": 0, "15-20": 0, ">20": 0},
    }

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task("Generating clips", total=n_clips)

        for i, ambient_idx in enumerate(chosen_indices):
            ambient_path = ambient_paths[int(ambient_idx)]
            clip_seed = seed + i + 1
            clip_id = f"train_seed{clip_seed}_{ambient_path.stem}"
            out_path = out_dir / f"{clip_id}.wav"

            result = generate_c1_1_clip(
                ambient_path=ambient_path,
                out_path=out_path,
                seed=clip_seed,
                priors=priors,
                biological_library_root=biological_library_root,
            )
            manifest = result["manifest"]

            n_sources = result["n_sources_realized"]
            n_bios = result["n_biologicals_realized"]
            if result["negative_label"]:
                summary_stats["negatives"] += 1
            else:
                summary_stats["positives_by_n_sources"][n_sources] = (
                    summary_stats["positives_by_n_sources"].get(n_sources, 0) + 1
                )
            summary_stats["total_lines"] += len(manifest.lines)
            summary_stats["total_biologicals"] += n_bios

            for line in manifest.lines:
                if not line.snr_curve_db:
                    continue
                mean_snr = float(np.mean(line.snr_curve_db))
                if mean_snr < 5:
                    summary_stats["snr_buckets"]["<5"] += 1
                elif mean_snr < 10:
                    summary_stats["snr_buckets"]["5-10"] += 1
                elif mean_snr < 15:
                    summary_stats["snr_buckets"]["10-15"] += 1
                elif mean_snr < 20:
                    summary_stats["snr_buckets"]["15-20"] += 1
                else:
                    summary_stats["snr_buckets"][">20"] += 1

            clip_summaries.append({
                "clip_id": clip_id,
                "wav_relative_path": result["wav_path"].relative_to(out_dir).as_posix(),
                "manifest_relative_path": (
                    result["manifest_path"].relative_to(out_dir).as_posix()
                ),
                "audit_relative_path": (
                    result["audit_path"].relative_to(out_dir).as_posix()
                ),
                "negative": result["negative_label"],
                "n_sources_realized": n_sources,
                "n_biologicals_realized": n_bios,
                "n_lines": len(manifest.lines),
                "seed": clip_seed,
                "ambient_compound_key": f"{ambient_path.parent.name}/{ambient_path.stem}",
            })

            progress.update(task, advance=1)

    dataset_manifest = {
        "dataset_id": out_dir.name,
        "n_clips": n_clips,
        "seed": seed,
        "ambient_dir": str(ambient_dir),
        "biological_library_root": str(biological_library_root),
        "split_manifest": str(split_manifest) if split_manifest else None,
        "split_partition": split_partition,
        "priors": {
            "n_harmonics_choices": list(priors.n_harmonics_choices),
            "harmonic_decay_range": list(priors.harmonic_decay_range),
            "total_persistence_log_range": list(priors.total_persistence_log_range),
            "drift_rate_std_hz_per_s": priors.drift_rate_std_hz_per_s,
        },
        "summary_stats": summary_stats,
        "clips": clip_summaries,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2)
    )

    CONSOLE.print(f"\n[green]Done. {n_clips} clips written to {out_dir}[/green]")
    CONSOLE.print("\n[cyan]Summary:[/cyan]")
    CONSOLE.print(
        f"  Negatives: {summary_stats['negatives']}/{n_clips} "
        f"({summary_stats['negatives'] / n_clips * 100:.1f}%)"
    )
    CONSOLE.print(
        f"  Positives by n_sources: {summary_stats['positives_by_n_sources']}"
    )
    CONSOLE.print(f"  Total tonal lines: {summary_stats['total_lines']}")
    CONSOLE.print(f"  Total biological overlays: {summary_stats['total_biologicals']}")
    CONSOLE.print(f"  Line SNR distribution: {summary_stats['snr_buckets']}")


if __name__ == "__main__":
    main()