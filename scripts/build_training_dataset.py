"""Bulk synthetic dataset generator for C3 ML detector training.

Loops generate_c1_1_clip across N clips with the full A1 §3.3 parameter sweep
(default TonalParameterPriors) + DCLDE biological confuser overlay (default
BiologicalInjectionPriors). No per-clip PNGs — training doesn't need them.

Writes per-clip WAV + truth manifest + audit sidecar into out_dir, plus a
top-level manifest.json listing all clips for reproducibility (Sprint 3
SplitManifest pattern).
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

from fathom.synthetic import generate_c1_1_clip

CONSOLE = Console()


def _list_ambient_wavs(ambient_dir: Path) -> list[Path]:
    return sorted(ambient_dir.rglob("*.wav"))


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
def main(
    ambient_dir: Path,
    out_dir: Path,
    n_clips: int,
    seed: int,
    biological_library_root: Path,
) -> None:
    """Generate a bulk synthetic training dataset (full A1 sweep + bios)."""
    ambient_paths = _list_ambient_wavs(ambient_dir)
    if not ambient_paths:
        raise click.UsageError(f"no .wav files under {ambient_dir}")
    CONSOLE.print(
        f"[cyan]Ambient sources: {len(ambient_paths)} files under {ambient_dir}[/cyan]"
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
            })

            progress.update(task, advance=1)

    dataset_manifest = {
        "dataset_id": "training_dataset_v1",
        "n_clips": n_clips,
        "seed": seed,
        "ambient_dir": str(ambient_dir),
        "biological_library_root": str(biological_library_root),
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