"""Standalone Tier-1 / Tier-2 evaluation for a trained ML-detector checkpoint.

Reads a training-run directory's config.json (for seed, architecture,
data_dir, val_fraction), reconstructs the val set, loads a checkpoint
(best.pt or last.pt), runs evaluate_model, and writes
tier1_metrics{_b<bin>_c<class>}.json.

Use when:
  - A training run is cancelled early but `best.pt` exists
  - Re-evaluating a finished run at different thresholds
  - Evaluating against an external val set (Sprint 5 Tier-2 mode)

Sprint 5 additions (2026-05-15):
  - --val-data-dir: optional external val dataset (Sprint 5 ratio-sweep
    checkpoints get evaluated against data/tier2_val_v2). When set,
    --val-fraction is ignored and ALL clips under --val-data-dir are
    used as val.
  - --bin-threshold + --class-threshold: pass-through to evaluate_model.
    Defaults (0.5/0.5) match Sprint 4. Output filename encodes the
    thresholds when non-default so sweeps don't clobber each other.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import torch
from rich.console import Console

from fathom.detection.ml_data import (
    PatchExtractionConfig,
    SyntheticPatchDataset,
    default_lofar_config,
)
from fathom.detection.ml_eval import evaluate_model, print_eval_summary
from fathom.detection.ml_train import build_model

CONSOLE = Console()


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _clip_level_train_val_split(
    clip_paths: list[Path], val_fraction: float, seed: int,
) -> tuple[list[Path], list[Path]]:
    """Mirror of scripts/train_ml_detector.py: clip-level split, seed-locked
    so train/val sets reproduce exactly."""
    rng = np.random.default_rng(seed)
    n_total = len(clip_paths)
    n_val = max(1, int(n_total * val_fraction))
    indices = rng.permutation(n_total)
    val_set = set(int(i) for i in indices[:n_val].tolist())
    train_paths = [p for i, p in enumerate(clip_paths) if i not in val_set]
    val_paths = [p for i, p in enumerate(clip_paths) if i in val_set]
    return train_paths, val_paths


@click.command()
@click.option(
    "--run-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Training run directory containing config.json and the checkpoint.",
)
@click.option(
    "--checkpoint",
    default="best.pt",
    help="Checkpoint filename within run-dir (best.pt or last.pt).",
)
@click.option(
    "--device",
    default="auto",
    help="auto|cpu|mps|cuda",
)
@click.option(
    "--unet-base-channels",
    type=int,
    default=64,
    help="U-Net base channels (must match training). 64=default, 32=smaller ablation. "
         "Workaround until config.json records this field — set to whatever the run used.",
)
@click.option(
    "--val-data-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional external validation dataset (Sprint 5 Tier-2 mode: "
         "data/tier2_val_v2). When provided, --val-fraction is ignored and "
         "ALL clips under this dir are used as val. Sprint 4 reproducibility: "
         "omit this flag and the script reconstructs the clip-level val split "
         "from config.json's data_dir + val_fraction.",
)
@click.option(
    "--bin-threshold",
    type=float,
    default=0.5,
    help="Mask/heatmap bin activation threshold for predicted-line extraction. "
         "0.5 = Sprint 4 default. Lower → more permissive predictions "
         "(more recall, more FP); higher → more conservative.",
)
@click.option(
    "--class-threshold",
    type=float,
    default=0.5,
    help="ResNet binary class-head threshold (ignored for U-Net mask-only "
         "predictions). 0.5 = Sprint 4 default.",
)
def main(
    run_dir: Path,
    checkpoint: str,
    device: str,
    unet_base_channels: int,
    val_data_dir: Path | None,
    bin_threshold: float,
    class_threshold: float,
) -> None:
    """Run Tier-1/Tier-2 evaluation on a trained checkpoint and write metrics JSON."""
    logging.getLogger("fathom.detection.ml_data").setLevel(logging.ERROR)

    config = json.loads((run_dir / "config.json").read_text())
    architecture = config["architecture"]
    seed = config["seed"]
    val_fraction = config["val_fraction"]
    data_dir = Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parents[1] / data_dir

    device_obj = _autodetect_device() if device == "auto" else torch.device(device)
    target_mode = "heatmap" if architecture == "resnet18" else "mask"

    if val_data_dir is not None:
        val_paths = sorted(val_data_dir.glob("*.wav"))
        if not val_paths:
            raise click.UsageError(f"no .wav files under {val_data_dir}")
        val_source_desc = f"external {val_data_dir} ({len(val_paths)} clips)"
    else:
        clip_paths = sorted(data_dir.glob("*.wav"))
        if not clip_paths:
            raise click.UsageError(f"no .wav files under {data_dir}")
        _train_paths, val_paths = _clip_level_train_val_split(
            clip_paths, val_fraction, seed,
        )
        val_source_desc = (
            f"clip-level split of {data_dir} (seed={seed}, "
            f"val_fraction={val_fraction}): {len(val_paths)} clips"
        )

    CONSOLE.print(
        f"[cyan]Architecture:[/cyan] {architecture}  "
        f"[cyan]target_mode:[/cyan] {target_mode}  "
        f"[cyan]device:[/cyan] {device_obj}"
    )
    CONSOLE.print(f"[cyan]Val source:[/cyan] {val_source_desc}")
    CONSOLE.print(
        f"[cyan]Thresholds:[/cyan] class={class_threshold}, bin={bin_threshold}"
    )

    val_ds = SyntheticPatchDataset(
        clip_paths=val_paths,
        lofar_config=default_lofar_config(),
        patch_config=PatchExtractionConfig(
            patch_size=256, stride=128, target_mode=target_mode,
        ),
    )
    CONSOLE.print(
        f"[cyan]Val:   {len(val_ds)} patches across "
        f"{len(val_ds._clip_entries)} usable clips[/cyan]"
    )

    torch.manual_seed(seed)
    model = build_model(
        architecture,
        num_freq_bins=256,
        unet_base_channels=unet_base_channels,
    ).to(device_obj)
    ckpt_path = run_dir / checkpoint
    state = torch.load(str(ckpt_path), map_location=device_obj)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    CONSOLE.print(f"[green]Loaded checkpoint: {ckpt_path}[/green]")

    CONSOLE.print("\n[cyan]Running patch-level evaluation...[/cyan]")
    eval_metrics = evaluate_model(
        model,
        val_ds,
        device_obj,
        architecture,
        class_threshold=class_threshold,
        bin_threshold=bin_threshold,
    )
    print_eval_summary(eval_metrics, console=CONSOLE)

    # Output filename: encode threshold + checkpoint variants so sweeps don't clobber
    is_default_thresholds = (class_threshold == 0.5 and bin_threshold == 0.5)
    is_best = checkpoint == "best.pt"
    if is_default_thresholds and is_best:
        out_name = "tier1_metrics.json"
    else:
        parts = ["tier1_metrics"]
        if not is_best:
            parts.append(Path(checkpoint).stem)
        if not is_default_thresholds:
            parts.append(f"b{bin_threshold:.2f}_c{class_threshold:.2f}")
        out_name = "_".join(parts) + ".json"

    out_path = run_dir / out_name
    out_path.write_text(json.dumps({
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(ckpt_path),
        "val_source": val_source_desc,
        "class_threshold": class_threshold,
        "bin_threshold": bin_threshold,
        **eval_metrics,
    }, indent=2))
    CONSOLE.print(f"\n[green]metrics: {out_path}[/green]")


if __name__ == "__main__":
    main()