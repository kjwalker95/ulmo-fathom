"""Sprint 6 Cluster B.6 + B.7 — ensemble-mean evaluation + diversity check.

Loads N U-Net member checkpoints, computes ensemble-mean predictions via
sigmoid-space averaging (Lakshminarayanan et al. 2017), evaluates the
ensemble-mean at bin_threshold against tier2_val_v2, and measures member
diversity via pairwise cosine similarity of per-pixel predictions.

Sigmoid-space averaging is the load-bearing implementation detail. C5
bimodal-saturation means individual member sigmoids saturate at {~0, ~1};
member disagreement shows up as intermediate ensemble-mean values (e.g.,
3 members at ~0.98 + 2 at ~0.02 yields mean ~0.59), populating the empty
reliability-diagram mid-bins. Logit-space averaging would defeat this
because pre-sigmoid logits don't carry the same monotone mapping when
individual members live in the saturated tails.

Per-pixel averaging happens BEFORE any thresholding, connected-component
extraction, or line post-processing (team review 2026-05-22): all members
tile the same gram identically with the same stride/patch_size; the
ensemble mask is the element-wise mean of per-pixel sigmoid mask tensors;
extract_predicted_lines_mask then runs on the averaged mask.

Outputs (under <output-dir>/):
  ensemble_eval.json   per-cell ensemble-mean tier1 metrics dict
  diversity.json       pairwise cosine sim + homogeneity verdict
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from fathom.detection.ml_data import (
    PatchExtractionConfig,
    SyntheticPatchDataset,
    default_lofar_config,
)
from fathom.detection.ml_eval import (
    SNR_BUCKETS,
    _snr_bucket_label,
    extract_predicted_lines_mask,
    extract_truth_lines_for_patch,
    hungarian_match,
    print_eval_summary,
)
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


def _run_ensemble(
    models: list[torch.nn.Module],
    dataset: SyntheticPatchDataset,
    device: torch.device,
    bin_threshold: float,
    iou_threshold: float,
) -> tuple[dict, dict]:
    """One pass through val patches: ensemble-mean prediction + diversity stats."""
    stft = dataset.lofar_config.stft
    full_freqs = np.fft.rfftfreq(stft.n_fft, d=1.0 / stft.sample_rate)
    band_mask = (
        (full_freqs >= dataset.lofar_config.freq_min_hz)
        & (full_freqs <= dataset.lofar_config.freq_max_hz)
    )
    gram_freqs = full_freqs[band_mask]
    freq_resolution_hz = stft.sample_rate / stft.n_fft
    frame_duration_s = stft.hop_length / stft.sample_rate

    n_members = len(models)
    bucket_tp: dict[str, int] = {label: 0 for _, _, label in SNR_BUCKETS}
    bucket_fn: dict[str, int] = {label: 0 for _, _, label in SNR_BUCKETS}
    total_tp = 0
    total_fp = 0
    total_truth = 0

    cos_sim_sums = np.zeros((n_members, n_members), dtype=np.float64)
    n_patches = 0

    clip_frame_times: dict[int, np.ndarray] = {}
    ps = dataset.patch_config.patch_size

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

        # Per-member sigmoid masks (SIGMOID space; logit-space averaging
        # would defeat the bimodal-saturation resolution mechanism).
        # UNetDetector.forward returns (B, H, W) with channel dim already
        # squeezed, so mask_logits[0] is (H, W).
        member_masks: list[np.ndarray] = []
        with torch.no_grad():
            for model in models:
                mask_logits = model(patch_t)
                mask_probs = torch.sigmoid(mask_logits[0]).cpu().numpy()
                member_masks.append(mask_probs)

        # Per-pixel ensemble mean — before any thresholding or extraction.
        ensemble_mask = np.stack(member_masks, axis=0).mean(axis=0)

        # Diversity: pairwise cosine sim of flattened sigmoid masks.
        flat = np.stack([m.ravel() for m in member_masks], axis=0)
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms_safe = np.where(norms > 1e-12, norms, 1.0)
        flat_norm = flat / norms_safe
        cos_sim = flat_norm @ flat_norm.T
        cos_sim_sums += cos_sim
        n_patches += 1

        truth_lines = extract_truth_lines_for_patch(
            entry.manifest.lines, gram_freqs, frame_times,
            f_start, t_start, ps,
        )
        total_truth += len(truth_lines)

        # Extract predicted lines from ENSEMBLE-MEAN mask.
        patch_t_start_s = float(frame_times[t_start])
        predicted = extract_predicted_lines_mask(
            ensemble_mask,
            patch_freq_axis_hz=gram_freqs[f_start:f_start + ps],
            patch_t_start_s=patch_t_start_s,
            patch_frame_duration_s=frame_duration_s,
            bin_threshold=bin_threshold,
        )

        matches, unmatched_pred, unmatched_truth = hungarian_match(
            predicted, truth_lines, freq_resolution_hz, iou_threshold,
        )
        total_tp += len(matches)
        total_fp += len(unmatched_pred)
        for _, truth_idx, _ in matches:
            label = _snr_bucket_label(truth_lines[truth_idx].peak_snr_db)
            bucket_tp[label] += 1
        for ti in unmatched_truth:
            label = _snr_bucket_label(truth_lines[ti].peak_snr_db)
            bucket_fn[label] += 1

    buckets_out: dict[str, dict] = {}
    for _, _, label in SNR_BUCKETS:
        tp = bucket_tp[label]
        fn = bucket_fn[label]
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        buckets_out[label] = {
            "tp": tp, "fn": fn, "n_truth": tp + fn, "recall": recall,
        }

    overall_precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0 else float("nan")
    )
    overall_recall = (
        total_tp / total_truth if total_truth > 0 else float("nan")
    )
    overall_f1 = (
        2 * overall_precision * overall_recall
        / (overall_precision + overall_recall)
        if overall_precision + overall_recall > 0 else float("nan")
    )

    gate_tp = sum(
        bucket_tp[lab] for _, _, lab in SNR_BUCKETS
        if lab in ("8-12", "12-20", ">=20")
    )
    gate_fn = sum(
        bucket_fn[lab] for _, _, lab in SNR_BUCKETS
        if lab in ("8-12", "12-20", ">=20")
    )
    gate_n = gate_tp + gate_fn
    gate_recall = gate_tp / gate_n if gate_n > 0 else float("nan")

    ensemble_metrics = {
        "buckets": buckets_out,
        "overall": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "tp": total_tp,
            "fp": total_fp,
            "total_truth": total_truth,
        },
        "acceptance_gate": {
            "snr_threshold_db": 8.0,
            "target_recall": 0.80,
            "actual_recall": gate_recall,
            "tp": gate_tp,
            "fn": gate_fn,
            "n_truth": gate_n,
            "passed": gate_recall >= 0.80 if gate_n > 0 else False,
        },
    }

    pairwise_mean = cos_sim_sums / max(n_patches, 1)
    upper_mask = np.triu(np.ones((n_members, n_members), dtype=bool), k=1)
    pairwise_values = pairwise_mean[upper_mask]
    high_homogeneity_threshold = 0.95
    diversity_stats = {
        "n_patches": n_patches,
        "n_members": n_members,
        "pairwise_cosine_sim_mean": float(pairwise_values.mean()),
        "pairwise_cosine_sim_median": float(np.median(pairwise_values)),
        "pairwise_cosine_sim_min": float(pairwise_values.min()),
        "pairwise_cosine_sim_max": float(pairwise_values.max()),
        "pairwise_cosine_sim_matrix": pairwise_mean.tolist(),
        "high_homogeneity_threshold": high_homogeneity_threshold,
        "homogeneous": bool(pairwise_values.mean() > high_homogeneity_threshold),
    }

    return ensemble_metrics, diversity_stats


@click.command()
@click.option(
    "--checkpoints", "checkpoints", multiple=True,
    type=click.Path(exists=True, path_type=Path), required=True,
    help="Member checkpoint paths (best.pt). Pass --checkpoints once per member.",
)
@click.option(
    "--val-data-dir",
    type=click.Path(exists=True, path_type=Path), required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path), required=True,
)
@click.option("--bin-threshold", type=float, default=0.001)
@click.option("--iou-threshold", type=float, default=0.1)
@click.option("--device", type=str, default="auto")
@click.option("--unet-base-channels", type=int, default=64)
def main(
    checkpoints: tuple[Path, ...],
    val_data_dir: Path,
    output_dir: Path,
    bin_threshold: float,
    iou_threshold: float,
    device: str,
    unet_base_channels: int,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = _autodetect_device() if device == "auto" else torch.device(device)

    CONSOLE.print(f"[cyan]Members ({len(checkpoints)}):[/cyan]")
    for cp in checkpoints:
        CONSOLE.print(f"  {cp}")
    CONSOLE.print(
        f"[cyan]Val: {val_data_dir}  bin_threshold={bin_threshold}  "
        f"device={device_obj}[/cyan]"
    )

    val_paths = sorted(Path(val_data_dir).glob("*.wav"))
    if not val_paths:
        raise click.UsageError(f"no .wav files under {val_data_dir}")
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

    CONSOLE.print(
        "\n[cyan]Running ensemble eval (sigmoid-space averaging)...[/cyan]"
    )
    ensemble_metrics, diversity_stats = _run_ensemble(
        models, val_ds, device_obj, bin_threshold, iou_threshold,
    )

    CONSOLE.print("\n[bold cyan]Ensemble-mean Tier-1 Evaluation[/bold cyan]")
    print_eval_summary(ensemble_metrics, console=CONSOLE)

    CONSOLE.print(
        "\n[bold cyan]Member diversity (pairwise cosine sim of sigmoid masks)[/bold cyan]"
    )
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("metric", justify="left")
    table.add_column("value", justify="right")
    table.add_row(
        "Mean",   f"{diversity_stats['pairwise_cosine_sim_mean']:.4f}",
    )
    table.add_row(
        "Median", f"{diversity_stats['pairwise_cosine_sim_median']:.4f}",
    )
    table.add_row(
        "Min",    f"{diversity_stats['pairwise_cosine_sim_min']:.4f}",
    )
    table.add_row(
        "Max",    f"{diversity_stats['pairwise_cosine_sim_max']:.4f}",
    )
    table.add_row(
        f"Homogeneous (mean > {diversity_stats['high_homogeneity_threshold']:.2f})?",
        str(diversity_stats["homogeneous"]),
    )
    CONSOLE.print(table)

    ensemble_out = {
        "computed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoints": [str(p) for p in checkpoints],
        "val_data_dir": str(val_data_dir),
        "bin_threshold": bin_threshold,
        "iou_threshold": iou_threshold,
        "ensemble_mean": ensemble_metrics,
    }
    ensemble_path = output_dir / "ensemble_eval.json"
    ensemble_path.write_text(json.dumps(ensemble_out, indent=2))
    CONSOLE.print(f"\n[green]Wrote {ensemble_path}[/green]")

    diversity_out = {
        "computed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoints": [str(p) for p in checkpoints],
        **diversity_stats,
    }
    diversity_path = output_dir / "diversity.json"
    diversity_path.write_text(json.dumps(diversity_out, indent=2))
    CONSOLE.print(f"[green]Wrote {diversity_path}[/green]")


if __name__ == "__main__":
    main()
