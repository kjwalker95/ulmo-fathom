"""Raw calibration measurement for ML detector checkpoints (Cluster C5).

PCD v4 §15.2 Gate 2 baseline. Measures patch-level binary calibration of
single trained U-Net detectors before any ensemble or conformal wrapping.
Sprint 6 targets ECE < 0.05 after deep ensemble + conformal; this baseline
number tells us how much the wrapping has to do.

Patch-level binary calibration:
  - predicted_score = max(sigmoid(mask_logits)) — model's most confident
    pixel that the patch contains a line
  - truth label = (patch contains any truth line per the binary_label)
  - Bin predicted_score into 10 equal-width bins; for each bin compute
    observed positive rate.
  - ECE = sum(bin_weight * |bin_mean_pred - bin_observed_rate|)
  - Reliability diagram = observed rate vs mean predicted per bin.

ResNet (heatmap head) path uses the class-head sigmoid instead of mask-max.

Sprint 5 Cluster C5 (2026-05-16).
"""
from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from torch.utils.data import DataLoader

from fathom.detection.ml_data import (
    PatchExtractionConfig,
    SyntheticPatchDataset,
    default_lofar_config,
)
from fathom.detection.ml_train import build_model

CONSOLE = Console()


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _patch_scores(
    model,
    patch: torch.Tensor,
    architecture: str,
) -> torch.Tensor:
    """Compute per-patch confidence score in [0, 1].

    - U-Net: max(sigmoid(mask)) across pixels.
    - ResNet (dual head): sigmoid of class-head logit.
    """
    if architecture == "unet":
        logits = model(patch)
        probs = torch.sigmoid(logits).squeeze(1)
        return probs.reshape(probs.shape[0], -1).max(dim=1).values
    if architecture == "resnet18":
        class_logits, _ = model(patch)
        return torch.sigmoid(class_logits).squeeze(-1)
    raise ValueError(f"unknown architecture {architecture!r}")


def _measure_calibration(
    *,
    model,
    val_ds,
    device: torch.device,
    architecture: str,
    batch_size: int = 64,
    n_bins: int = 10,
) -> dict:
    """Run inference over val_ds, compute patch-level binary calibration."""
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    pred_scores: list[float] = []
    truth_labels: list[float] = []

    model.eval()
    with torch.no_grad():
        for patch, binary_label, _target in loader:
            patch = patch.to(device)
            scores = _patch_scores(model, patch, architecture)
            pred_scores.extend(scores.cpu().tolist())
            truth_labels.extend(binary_label.cpu().tolist())

    pred = np.array(pred_scores)
    truth = np.array(truth_labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    ece = 0.0
    n_total = len(pred)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (pred >= lo) & (pred < hi)
        else:
            mask = (pred >= lo) & (pred <= hi)
        n_in = int(mask.sum())
        if n_in == 0:
            bins.append({
                "bin_idx": i,
                "lower": float(lo),
                "upper": float(hi),
                "n_samples": 0,
                "mean_predicted": None,
                "observed_positive_rate": None,
                "abs_calibration_gap": None,
            })
            continue
        mean_pred = float(pred[mask].mean())
        obs_rate = float(truth[mask].mean())
        gap = abs(mean_pred - obs_rate)
        ece += (n_in / n_total) * gap
        bins.append({
            "bin_idx": i,
            "lower": float(lo),
            "upper": float(hi),
            "n_samples": n_in,
            "mean_predicted": mean_pred,
            "observed_positive_rate": obs_rate,
            "abs_calibration_gap": gap,
        })

    nonempty = [b for b in bins if b["n_samples"]]
    overconfident = sum(
        1 for b in nonempty
        if b["mean_predicted"] > b["observed_positive_rate"]
    )
    return {
        "n_patches": int(n_total),
        "n_positives": int(truth.sum()),
        "n_bins": n_bins,
        "ece": float(ece),
        "overconfidence_bin_fraction": (
            overconfident / len(nonempty) if nonempty else 0.0
        ),
        "bins": bins,
    }


def _save_reliability_diagram(metrics: dict, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    xs: list[float] = []
    ys: list[float] = []
    sizes: list[int] = []
    for b in metrics["bins"]:
        if b["n_samples"] == 0:
            continue
        xs.append(b["mean_predicted"])
        ys.append(b["observed_positive_rate"])
        sizes.append(b["n_samples"])
    if xs:
        sizes_arr = np.array(sizes, dtype=float)
        sizes_arr = 20.0 + 200.0 * (sizes_arr / sizes_arr.max())
        ax.scatter(xs, ys, s=sizes_arr, alpha=0.7, edgecolor="black")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfect calibration")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed positive rate (bin)")
    ax.set_title(
        f"{title}\nECE = {metrics['ece']:.4f}  |  "
        f"N = {metrics['n_patches']}  positives = {metrics['n_positives']}"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


@click.command()
@click.option(
    "--sweep-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Sweep root containing the winning-ratio cells.",
)
@click.option(
    "--ratio",
    type=float,
    default=0.75,
    help="Winning ratio (per Cluster C3 selection).",
)
@click.option(
    "--seeds",
    type=str,
    default="20260512,20260513,20260514",
    help="Comma-separated seeds to evaluate.",
)
@click.option(
    "--val-data-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Validation dataset (Sprint 5: data/tier2_val_v2).",
)
@click.option("--device", type=str, default="auto", help="auto|cpu|mps|cuda")
@click.option("--unet-base-channels", type=int, default=64)
@click.option("--architecture", type=str, default="unet")
def main(
    sweep_dir: Path,
    ratio: float,
    seeds: str,
    val_data_dir: Path,
    device: str,
    unet_base_channels: int,
    architecture: str,
) -> None:
    """Measure raw single-model calibration (Cluster C5)."""
    logging.getLogger("fathom.detection.ml_data").setLevel(logging.ERROR)

    seed_list = [int(s) for s in seeds.split(",")]
    device_obj = _autodetect_device() if device == "auto" else torch.device(device)
    target_mode = "heatmap" if architecture == "resnet18" else "mask"

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

    per_seed_results: list[dict] = []
    for seed in seed_list:
        cell = sweep_dir / f"unet_seed{seed}_ratio{ratio:.2f}"
        ckpt = cell / "best.pt"
        if not ckpt.exists():
            CONSOLE.print(f"[yellow]skip[/yellow] {cell.name}: no best.pt")
            continue
        state = torch.load(str(ckpt), map_location=device_obj)
        state_dict = (
            state.get("model_state_dict", state) if isinstance(state, dict) else state
        )
        model = build_model(
            architecture,
            num_freq_bins=256,
            unet_base_channels=unet_base_channels,
        ).to(device_obj)
        model.load_state_dict(state_dict)

        CONSOLE.print(f"\n[cyan]Measuring calibration:[/cyan] {cell.name}")
        metrics = _measure_calibration(
            model=model,
            val_ds=val_ds,
            device=device_obj,
            architecture=architecture,
            batch_size=64,
            n_bins=10,
        )
        metrics["seed"] = seed
        metrics["ratio"] = ratio
        metrics["measured_at_utc"] = datetime.now(timezone.utc).isoformat()
        metrics["val_data_dir"] = str(val_data_dir)
        metrics["checkpoint"] = str(ckpt)
        metrics["architecture"] = architecture
        metrics["unet_base_channels"] = unet_base_channels

        json_path = cell / "calibration_metrics.json"
        json_path.write_text(json.dumps(metrics, indent=2))
        png_path = cell / "reliability_diagram.png"
        _save_reliability_diagram(metrics, png_path, f"{cell.name}")

        CONSOLE.print(
            f"  ECE={metrics['ece']:.4f}  "
            f"N={metrics['n_patches']}  "
            f"positives={metrics['n_positives']}  "
            f"overconfident_bin_frac={metrics['overconfidence_bin_fraction']:.2f}"
        )
        per_seed_results.append(metrics)

    if not per_seed_results:
        raise click.UsageError("no cells measured (no best.pt found)")

    eces = [m["ece"] for m in per_seed_results]
    summary = {
        "n_seeds_measured": len(per_seed_results),
        "ratio": ratio,
        "ece_mean": statistics.mean(eces),
        "ece_std": statistics.stdev(eces) if len(eces) > 1 else 0.0,
        "ece_per_seed": [
            {"seed": m["seed"], "ece": m["ece"]} for m in per_seed_results
        ],
        "sprint6_target_ece": 0.05,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = sweep_dir / f"calibration_summary_ratio{ratio:.2f}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    CONSOLE.print(f"\n[bold green]Cluster C5 summary:[/bold green]")
    CONSOLE.print(
        f"  ECE: {summary['ece_mean']:.4f} ± {summary['ece_std']:.4f} "
        f"(n_seeds={summary['n_seeds_measured']})"
    )
    CONSOLE.print(f"  Sprint 6 target: ECE < {summary['sprint6_target_ece']}")
    CONSOLE.print(f"  Summary: {summary_path}")
    CONSOLE.print(
        "  Per-cell artifacts: calibration_metrics.json + reliability_diagram.png"
    )


if __name__ == "__main__":
    main()