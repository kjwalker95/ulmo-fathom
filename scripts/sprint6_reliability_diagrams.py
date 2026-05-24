"""Sprint 6 Cluster C.2 — bimodal-saturation diagnosis via reliability diagrams.

Loads N U-Net ensemble checkpoints, runs val-set inference once, computes
per-patch per-member sigmoid masks, then evaluates calibration across multiple
(scoring function, patch aggregation) setups:

  Phase 1 - Per-member reliability at max_mean aggregation (N plots).
            Confirms / refutes individual-member bimodal saturation.
  Phase 2 - Ensemble-mean reliability at each of 3 aggregations (3 plots).
            Picks the winning aggregation for mean_prediction scoring.
  Phase 3 - Per-pixel non-mean scoring function distributions (3 histograms).
            predictive_entropy, mutual_information, member_disagreement_variance
            split by truth label (positive vs negative patch). Shows whether
            mid-range mass exists in each disagreement signal.

Outputs (under <output-dir>/):
  per_member_seed<S>_max_mean.png   N plots
  ensemble_<aggregation>.png         3 plots: max_mean, mean_max, peak_freq_band
  score_dist_<function>.png          3 histograms
  reliability_summary.json           ECE + occupancy for all setups
  reliability_summary.md             table for C.4 decision + commit message
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from fathom.calibration.ensemble import (
    compute_reliability_bins,
    mean_prediction,
    member_disagreement_variance,
    mutual_information,
    patch_confidence,
    predictive_entropy,
)
from fathom.detection.ml_data import (
    PatchExtractionConfig,
    SyntheticPatchDataset,
    default_lofar_config,
)
from fathom.detection.ml_eval import extract_truth_lines_for_patch
from fathom.detection.ml_train import build_model

CONSOLE = Console()


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_member(
    checkpoint: Path, unet_base_channels: int, device: torch.device
) -> torch.nn.Module:
    model = build_model(
        "unet", num_freq_bins=256, unet_base_channels=unet_base_channels,
    ).to(device)
    state = torch.load(str(checkpoint), map_location=device)
    state_dict = (
        state.get("model_state_dict", state) if isinstance(state, dict) else state
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _save_reliability_diagram(metrics: dict, out_path: Path, title: str) -> None:
    """Save a reliability scatter + diagonal + bin-population annotations."""
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
        for x, y, n in zip(xs, ys, sizes):
            ax.annotate(
                f"n={n}", (x, y), textcoords="offset points",
                xytext=(7, 0), fontsize=8,
            )
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfect calibration")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed positive rate (bin)")
    occupied = sum(1 for b in metrics["bins"] if b["n_samples"] > 0)
    ax.set_title(
        f"{title}\nECE={metrics['ece']:.4f}  occupied bins={occupied}/10  "
        f"N={metrics['n_patches']}  positives={metrics['n_positives']}"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


def _save_score_distribution(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
    function_name: str,
    out_path: Path,
) -> None:
    """Histogram of a per-patch scoring-function output split by truth label."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = 30
    ax.hist(
        scores_neg, bins=bins, alpha=0.6, label=f"negative patches (n={len(scores_neg)})",
        color="steelblue",
    )
    ax.hist(
        scores_pos, bins=bins, alpha=0.6, label=f"positive patches (n={len(scores_pos)})",
        color="firebrick",
    )
    ax.set_xlabel(f"per-patch {function_name} (max over pixels)")
    ax.set_ylabel("count")
    ax.set_title(f"Per-patch {function_name} distribution by truth label")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


def _collect_per_patch(
    models: list[torch.nn.Module],
    dataset: SyntheticPatchDataset,
    device: torch.device,
) -> dict:
    """Single inference pass: collect per-patch per-member sigmoid masks + all
    scoring-function scalars + binary truth labels.

    Returns a dict of numpy arrays each of shape (n_patches,) for the per-patch
    scalars, plus the binary truth labels.
    """
    stft = dataset.lofar_config.stft
    full_freqs = np.fft.rfftfreq(stft.n_fft, d=1.0 / stft.sample_rate)
    band_mask = (
        (full_freqs >= dataset.lofar_config.freq_min_hz)
        & (full_freqs <= dataset.lofar_config.freq_max_hz)
    )
    gram_freqs = full_freqs[band_mask]

    n_members = len(models)
    ps = dataset.patch_config.patch_size

    per_member_max_mean: list[list[float]] = [[] for _ in range(n_members)]
    ensemble_max_mean: list[float] = []
    ensemble_mean_max: list[float] = []
    ensemble_peak_freq_band: list[float] = []
    max_predictive_entropy: list[float] = []
    max_mutual_information: list[float] = []
    max_member_variance: list[float] = []
    truth_labels: list[int] = []

    clip_frame_times: dict[int, np.ndarray] = {}

    for patch_addr in dataset._patch_addresses:
        clip_idx = patch_addr.clip_idx
        entry = dataset._clip_entries[clip_idx]
        f_start = patch_addr.f_start
        t_start = patch_addr.t_start

        if clip_idx not in clip_frame_times:
            n_frames = entry.n_time_frames
            clip_frame_times[clip_idx] = (
                (np.arange(n_frames) * stft.hop_length + stft.window_length / 2.0)
                / stft.sample_rate
            )
        frame_times = clip_frame_times[clip_idx]

        gram = dataset._get_gram(clip_idx, entry)
        patch_np = gram.normalized_power_db[
            f_start:f_start + ps, t_start:t_start + ps
        ].astype(np.float32)
        patch_t = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(0).to(device)

        member_masks: list[np.ndarray] = []
        with torch.no_grad():
            for model in models:
                mask_logits = model(patch_t)
                mask_probs = torch.sigmoid(mask_logits[0]).cpu().numpy()
                member_masks.append(mask_probs)
        masks_stack = np.stack(member_masks, axis=0)  # (N, H, W)

        # Per-member confidence at max_mean (each member's max sigmoid pixel)
        for m_idx, m in enumerate(member_masks):
            per_member_max_mean[m_idx].append(float(m.max()))

        # Ensemble-mean confidence at 3 aggregations
        ensemble_max_mean.append(patch_confidence(masks_stack, method="max_mean"))
        ensemble_mean_max.append(patch_confidence(masks_stack, method="mean_max"))
        ensemble_peak_freq_band.append(
            patch_confidence(masks_stack, method="peak_freq_band")
        )

        # Per-pixel scoring functions, reduced to per-patch scalar via max
        max_predictive_entropy.append(float(predictive_entropy(masks_stack).max()))
        max_mutual_information.append(float(mutual_information(masks_stack).max()))
        max_member_variance.append(
            float(member_disagreement_variance(masks_stack).max())
        )

        # Truth label: 1 if patch contains any truth line, else 0
        truth_lines = extract_truth_lines_for_patch(
            entry.manifest.lines, gram_freqs, frame_times,
            f_start, t_start, ps,
        )
        truth_labels.append(1 if truth_lines else 0)

    return {
        "per_member_max_mean": np.array(per_member_max_mean),  # (N, n_patches)
        "ensemble_max_mean": np.array(ensemble_max_mean),
        "ensemble_mean_max": np.array(ensemble_mean_max),
        "ensemble_peak_freq_band": np.array(ensemble_peak_freq_band),
        "max_predictive_entropy": np.array(max_predictive_entropy),
        "max_mutual_information": np.array(max_mutual_information),
        "max_member_variance": np.array(max_member_variance),
        "truth_labels": np.array(truth_labels),
    }


@click.command()
@click.option(
    "--checkpoints", "checkpoints", multiple=True,
    type=click.Path(exists=True, path_type=Path), required=True,
    help="Member checkpoint paths (best.pt). Pass once per member.",
)
@click.option(
    "--val-data-dir",
    type=click.Path(exists=True, path_type=Path), required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path), required=True,
)
@click.option("--device", type=str, default="auto")
@click.option("--unet-base-channels", type=int, default=64)
def main(
    checkpoints: tuple[Path, ...],
    val_data_dir: Path,
    output_dir: Path,
    device: str,
    unet_base_channels: int,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = _autodetect_device() if device == "auto" else torch.device(device)

    seeds = [cp.parent.name.split("_seed")[1].split("_")[0] for cp in checkpoints]
    CONSOLE.print(f"[cyan]Members ({len(checkpoints)}):[/cyan]")
    for cp in checkpoints:
        CONSOLE.print(f"  {cp}")
    CONSOLE.print(f"[cyan]Val: {val_data_dir}  device={device_obj}[/cyan]")

    val_paths = sorted(Path(val_data_dir).glob("*.wav"))
    val_ds = SyntheticPatchDataset(
        clip_paths=val_paths,
        lofar_config=default_lofar_config(),
        patch_config=PatchExtractionConfig(
            patch_size=256, stride=128, target_mode="mask",
        ),
    )
    CONSOLE.print(
        f"[cyan]Val: {len(val_ds)} patches across "
        f"{len(val_ds._clip_entries)} usable clips[/cyan]"
    )

    models = [_load_member(cp, unet_base_channels, device_obj) for cp in checkpoints]

    CONSOLE.print("\n[cyan]Running inference + collecting per-patch scores...[/cyan]")
    scalars = _collect_per_patch(models, val_ds, device_obj)
    truth = scalars["truth_labels"]
    CONSOLE.print(
        f"[cyan]Collected: {len(truth)} patches, {int(truth.sum())} positive[/cyan]"
    )

    # Phase 1: per-member reliability diagrams at max_mean
    summary: list[dict] = []
    for m_idx, seed in enumerate(seeds):
        scores = scalars["per_member_max_mean"][m_idx]
        bins = compute_reliability_bins(scores, truth, n_bins=10)
        out_png = output_dir / f"per_member_seed{seed}_max_mean.png"
        _save_reliability_diagram(
            bins, out_png, f"Member seed={seed} max_mean",
        )
        occupied = sum(1 for b in bins["bins"] if b["n_samples"] > 0)
        summary.append({
            "setup": f"member_seed{seed}_max_mean",
            "ece": bins["ece"],
            "occupied_bins": occupied,
            "overconfidence_fraction": bins["overconfidence_bin_fraction"],
        })

    # Phase 2: ensemble-mean reliability diagrams at 3 aggregations
    for agg in ("max_mean", "mean_max", "peak_freq_band"):
        scores = scalars[f"ensemble_{agg}"]
        bins = compute_reliability_bins(scores, truth, n_bins=10)
        out_png = output_dir / f"ensemble_{agg}.png"
        _save_reliability_diagram(bins, out_png, f"Ensemble mean - {agg}")
        occupied = sum(1 for b in bins["bins"] if b["n_samples"] > 0)
        summary.append({
            "setup": f"ensemble_{agg}",
            "ece": bins["ece"],
            "occupied_bins": occupied,
            "overconfidence_fraction": bins["overconfidence_bin_fraction"],
        })

    # Phase 3: per-pixel non-mean scoring function distributions
    pos_mask = truth.astype(bool)
    neg_mask = ~pos_mask
    for fn_name, key in [
        ("predictive_entropy", "max_predictive_entropy"),
        ("mutual_information", "max_mutual_information"),
        ("member_disagreement_variance", "max_member_variance"),
    ]:
        scores = scalars[key]
        out_png = output_dir / f"score_dist_{fn_name}.png"
        _save_score_distribution(
            scores[pos_mask], scores[neg_mask], fn_name, out_png,
        )

    # Pretty print + persist summary
    CONSOLE.print("\n[bold cyan]Reliability summary[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setup", justify="left")
    table.add_column("ECE", justify="right")
    table.add_column("Occupied bins / 10", justify="right")
    table.add_column("Overconfidence frac", justify="right")
    for row in summary:
        table.add_row(
            row["setup"],
            f"{row['ece']:.4f}",
            f"{row['occupied_bins']}",
            f"{row['overconfidence_fraction']:.2f}",
        )
    CONSOLE.print(table)

    # JSON + markdown outputs
    summary_path = output_dir / "reliability_summary.json"
    summary_path.write_text(json.dumps({
        "computed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoints": [str(p) for p in checkpoints],
        "val_data_dir": str(val_data_dir),
        "n_patches": int(len(truth)),
        "n_positives": int(truth.sum()),
        "sprint5_c5_baseline_ece_mean": 0.0746,
        "sprint5_c5_baseline_ece_std": 0.0101,
        "setups": summary,
    }, indent=2))
    CONSOLE.print(f"\n[green]Wrote {summary_path}[/green]")

    md_path = output_dir / "reliability_summary.md"
    md_lines = [
        "# Sprint 6 C.2 reliability summary",
        "",
        f"- Val: {val_data_dir}  ({len(truth)} patches, {int(truth.sum())} positive)",
        f"- Sprint 5 C5 single-model baseline: ECE = 0.0746 +/- 0.0101 (2-3 occupied bins)",
        "",
        "| Setup | ECE | Occupied bins / 10 | Overconfidence frac |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        md_lines.append(
            f"| {row['setup']} | {row['ece']:.4f} | {row['occupied_bins']} | "
            f"{row['overconfidence_fraction']:.2f} |"
        )
    md_lines.append("")
    md_lines.append(
        "C.4 acceptance: at least one setup should produce >=5 occupied bins "
        "AND ECE < 0.0746. Winning (function, aggregation) pair feeds Cluster D."
    )
    md_path.write_text("\n".join(md_lines) + "\n")
    CONSOLE.print(f"[green]Wrote {md_path}[/green]")


if __name__ == "__main__":
    main()