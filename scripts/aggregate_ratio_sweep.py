"""Aggregate Tier-2 metrics across the Cluster C2 ratio sweep.

Walks <sweep_dir>/unet_seed*_ratio*/, loads each cell's best.pt, evaluates
against an external Tier-2 val dataset at multiple bin_thresholds, then
aggregates across the 3 seeds per ratio to identify the winning (ratio,
threshold) combo.

Cluster C3 deliverable: artifacts/sprint5_ratio_sweep/aggregate.json plus
a summary table to stdout. Sprint5_Plan §C3 'U-minimum': max mean recall
@ SNR>=8 dB where the confidence interval does not overlap adjacent cells.

Sprint 5 Cluster C3 (2026-05-16).
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import click
import torch
from rich.console import Console
from rich.table import Table

from fathom.detection.ml_data import (
    PatchExtractionConfig,
    SyntheticPatchDataset,
    default_lofar_config,
)
from fathom.detection.ml_eval import evaluate_model
from fathom.detection.ml_train import build_model

CONSOLE = Console()
CELL_PATTERN = re.compile(r"^unet_seed(\d+)_ratio([\d.]+)$")


@dataclass(frozen=True)
class CellEvalResult:
    seed: int
    ratio: float
    threshold: float
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    n_truth: int
    recall_at_8db: float


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_cell(cell_dir: Path) -> tuple[int, float] | None:
    m = CELL_PATTERN.match(cell_dir.name)
    if not m:
        return None
    return int(m.group(1)), float(m.group(2))


def _load_model(
    cell_dir: Path,
    architecture: str,
    unet_base_channels: int,
    device: torch.device,
):
    state = torch.load(str(cell_dir / "best.pt"), map_location=device)
    state_dict = (
        state.get("model_state_dict", state) if isinstance(state, dict) else state
    )
    model = build_model(
        architecture,
        num_freq_bins=256,
        unet_base_channels=unet_base_channels,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _eval_cell_at_threshold(
    *,
    model,
    val_ds,
    architecture: str,
    device: torch.device,
    seed: int,
    ratio: float,
    threshold: float,
) -> CellEvalResult:
    metrics = evaluate_model(
        model,
        val_ds,
        device,
        architecture,
        class_threshold=0.5,
        bin_threshold=threshold,
    )
    overall = metrics["overall"]
    gate = metrics["acceptance_gate"]
    return CellEvalResult(
        seed=seed,
        ratio=ratio,
        threshold=threshold,
        f1=float(overall["f1"]),
        precision=float(overall["precision"]),
        recall=float(overall["recall"]),
        tp=int(overall["tp"]),
        fp=int(overall["fp"]),
        n_truth=int(overall["total_truth"]),
        recall_at_8db=float(gate["actual_recall"]),
    )


def _aggregate(results: list[CellEvalResult]) -> list[dict]:
    by_key: dict[tuple[float, float], list[CellEvalResult]] = {}
    for r in results:
        by_key.setdefault((r.ratio, r.threshold), []).append(r)

    aggregates: list[dict] = []
    for (ratio, threshold), seeds in sorted(by_key.items()):
        f1s = [s.f1 for s in seeds]
        recall8s = [s.recall_at_8db for s in seeds]
        fps = [s.fp for s in seeds]
        aggregates.append({
            "ratio": ratio,
            "threshold": threshold,
            "n_seeds": len(seeds),
            "f1_mean": statistics.mean(f1s),
            "f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
            "recall_at_8db_mean": statistics.mean(recall8s),
            "recall_at_8db_std": (
                statistics.stdev(recall8s) if len(recall8s) > 1 else 0.0
            ),
            "fp_mean": statistics.mean(fps),
            "fp_std": statistics.stdev(fps) if len(fps) > 1 else 0.0,
            "per_seed": [
                {
                    "seed": s.seed,
                    "f1": s.f1,
                    "precision": s.precision,
                    "recall": s.recall,
                    "recall_at_8db": s.recall_at_8db,
                    "tp": s.tp,
                    "fp": s.fp,
                    "n_truth": s.n_truth,
                }
                for s in sorted(seeds, key=lambda x: x.seed)
            ],
        })
    return aggregates


def _identify_winning(aggregates: list[dict]) -> dict:
    """Sprint5_Plan §C3 U-minimum: max mean recall@>=8dB where ±std CI doesn't
    overlap adjacent cells. The CI calc here is one-sigma (advisory); n=3
    seeds is too few for 95% CI bootstrap to mean much."""
    best = max(aggregates, key=lambda a: a["recall_at_8db_mean"])
    best_lo = best["recall_at_8db_mean"] - best["recall_at_8db_std"]
    overlaps = [
        a for a in aggregates
        if a is not best
        and (a["recall_at_8db_mean"] + a["recall_at_8db_std"]) >= best_lo
    ]
    return {
        "ratio": best["ratio"],
        "threshold": best["threshold"],
        "recall_at_8db_mean": best["recall_at_8db_mean"],
        "recall_at_8db_std": best["recall_at_8db_std"],
        "f1_mean": best["f1_mean"],
        "f1_std": best["f1_std"],
        "passes_80pct_gate": best["recall_at_8db_mean"] >= 0.80,
        "n_cells_with_overlapping_ci": len(overlaps),
        "overlapping_cells": [
            {
                "ratio": a["ratio"],
                "threshold": a["threshold"],
                "recall_at_8db_mean": a["recall_at_8db_mean"],
            }
            for a in overlaps
        ],
    }


def _print_summary_tables(aggregates: list[dict]) -> None:
    thresholds = sorted({a["threshold"] for a in aggregates})
    ratios = sorted({a["ratio"] for a in aggregates})
    by_key = {(a["ratio"], a["threshold"]): a for a in aggregates}

    CONSOLE.print(
        "\n[bold cyan]Recall @ SNR >= 8 dB (mean ± std across 3 seeds)[/bold cyan]"
    )
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ratio \\ thr")
    for t in thresholds:
        table.add_column(f"{t:g}", justify="right")
    for r in ratios:
        row = [f"{r:.2f}"]
        for t in thresholds:
            cell = by_key.get((r, t))
            row.append(
                "—" if cell is None
                else f"{cell['recall_at_8db_mean']:.3f}\n±{cell['recall_at_8db_std']:.3f}"
            )
        table.add_row(*row)
    CONSOLE.print(table)

    CONSOLE.print("\n[bold cyan]F1 (mean ± std across 3 seeds)[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ratio \\ thr")
    for t in thresholds:
        table.add_column(f"{t:g}", justify="right")
    for r in ratios:
        row = [f"{r:.2f}"]
        for t in thresholds:
            cell = by_key.get((r, t))
            row.append(
                "—" if cell is None
                else f"{cell['f1_mean']:.3f}\n±{cell['f1_std']:.3f}"
            )
        table.add_row(*row)
    CONSOLE.print(table)


@click.command()
@click.option(
    "--sweep-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Sweep root containing unet_seed*_ratio*/ subdirs.",
)
@click.option(
    "--val-data-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="External Tier-2 val dataset (data/tier2_val_v2).",
)
@click.option(
    "--thresholds",
    type=str,
    default="0.01,0.05,0.10,0.20,0.30,0.50",
    help="Comma-separated bin_thresholds to sweep per checkpoint.",
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="auto|cpu|mps|cuda",
)
@click.option(
    "--unet-base-channels",
    type=int,
    default=64,
    help="U-Net base channels (must match training).",
)
@click.option(
    "--architecture",
    type=str,
    default="unet",
    help="Architecture (currently only 'unet' supported here).",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON path (default: <sweep-dir>/aggregate.json).",
)
def main(
    sweep_dir: Path,
    val_data_dir: Path,
    thresholds: str,
    device: str,
    unet_base_channels: int,
    architecture: str,
    output: Path | None,
) -> None:
    """Aggregate Tier-2 metrics across the ratio sweep and identify the
    winning (ratio, threshold) combo (Sprint 5 Cluster C3)."""
    logging.getLogger("fathom.detection.ml_data").setLevel(logging.ERROR)

    threshold_list = [float(t.strip()) for t in thresholds.split(",") if t.strip()]
    device_obj = _autodetect_device() if device == "auto" else torch.device(device)
    target_mode = "heatmap" if architecture == "resnet18" else "mask"

    if output is None:
        output = sweep_dir / "aggregate.json"

    CONSOLE.print(f"[cyan]Sweep dir:[/cyan] {sweep_dir}")
    CONSOLE.print(f"[cyan]Val data:[/cyan] {val_data_dir}")
    CONSOLE.print(f"[cyan]Thresholds:[/cyan] {threshold_list}")
    CONSOLE.print(f"[cyan]Device:[/cyan] {device_obj}")

    val_paths = sorted(val_data_dir.glob("*.wav"))
    if not val_paths:
        raise click.UsageError(f"no .wav files under {val_data_dir}")
    val_ds = SyntheticPatchDataset(
        clip_paths=val_paths,
        lofar_config=default_lofar_config(),
        patch_config=PatchExtractionConfig(
            patch_size=256, stride=128, target_mode=target_mode,
        ),
    )
    CONSOLE.print(
        f"[cyan]Val patches:[/cyan] {len(val_ds)} across "
        f"{len(val_ds._clip_entries)} clips"
    )

    cell_dirs = sorted(
        d for d in sweep_dir.iterdir() if d.is_dir() and CELL_PATTERN.match(d.name)
    )
    CONSOLE.print(f"[cyan]Cells found:[/cyan] {len(cell_dirs)}")

    results: list[CellEvalResult] = []
    total_evals = len(cell_dirs) * len(threshold_list)
    eval_num = 0

    for cell_dir in cell_dirs:
        parsed = _parse_cell(cell_dir)
        if parsed is None:
            continue
        seed, ratio = parsed
        ckpt_path = cell_dir / "best.pt"
        if not ckpt_path.exists():
            CONSOLE.print(f"[yellow]skip:[/yellow] {cell_dir.name} (no best.pt)")
            continue

        CONSOLE.print(f"\n[cyan]Loading[/cyan] {cell_dir.name}")
        model = _load_model(cell_dir, architecture, unet_base_channels, device_obj)

        for thr in threshold_list:
            eval_num += 1
            CONSOLE.print(
                f"  [{eval_num}/{total_evals}] threshold={thr}", end=" "
            )
            result = _eval_cell_at_threshold(
                model=model,
                val_ds=val_ds,
                architecture=architecture,
                device=device_obj,
                seed=seed,
                ratio=ratio,
                threshold=thr,
            )
            results.append(result)
            CONSOLE.print(
                f"F1={result.f1:.3f} recall@8dB={result.recall_at_8db:.3f}"
            )

    aggregates = _aggregate(results)
    winning = _identify_winning(aggregates)

    payload = {
        "aggregation_built_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_dir": str(sweep_dir),
        "val_data_dir": str(val_data_dir),
        "thresholds_swept": threshold_list,
        "n_cells_evaluated": len(cell_dirs),
        "n_evals_total": len(results),
        "cells": aggregates,
        "winning_combo": winning,
    }
    output.write_text(json.dumps(payload, indent=2))
    CONSOLE.print(f"\n[green]Aggregate written: {output}[/green]")

    _print_summary_tables(aggregates)

    CONSOLE.print(f"\n[bold green]Winning combo:[/bold green]")
    CONSOLE.print(
        f"  ratio={winning['ratio']:.2f}  threshold={winning['threshold']:.2f}"
    )
    CONSOLE.print(
        f"  recall @ SNR>=8 dB: {winning['recall_at_8db_mean']:.3f} ± "
        f"{winning['recall_at_8db_std']:.3f}"
    )
    CONSOLE.print(
        f"  F1: {winning['f1_mean']:.3f} ± {winning['f1_std']:.3f}"
    )
    CONSOLE.print(
        f"  Passes 80% gate: "
        f"{'[green]YES[/green]' if winning['passes_80pct_gate'] else '[yellow]NO[/yellow]'}"
    )
    CONSOLE.print(
        f"  Cells with overlapping ±std CI: "
        f"{winning['n_cells_with_overlapping_ci']}"
    )


if __name__ == "__main__":
    main()