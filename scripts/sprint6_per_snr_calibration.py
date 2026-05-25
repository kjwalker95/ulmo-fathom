"""Sprint 6 Cluster E.3 - per-SNR-bucket calibration on Tier-2 val.

Loads N U-Net ensemble checkpoints, runs val inference, computes:
  - Per-patch max_mean ensemble confidence (C.4 winner)
  - Per-patch binary truth label (any line in patch)
  - Per-patch peak SNR (max peak_snr_db across truth lines in patch)

Bins patches by SNR {<0, 0-5, 5-8, 8-12, 12-20, >=20} and reports
per-bucket reliability bins + ECE. The n=10 rule (team review 2026-05-25):
buckets with n_patches < 10 are noise; adjacent-collapse them. Specifically:
  if either <0 or 0-5 has n<10, merge into <5
  if either 12-20 or >=20 has n<10, merge into >=12

Outputs (under <output-dir>/):
  per_snr_calibration.json  raw + merged per-bucket ECE
  per_snr_calibration.png   bar chart with n annotated per bar
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

from fathom.calibration.ensemble import compute_reliability_bins, patch_confidence
from fathom.detection.ml_data import (
    PatchExtractionConfig,
    SyntheticPatchDataset,
    default_lofar_config,
)
from fathom.detection.ml_eval import extract_truth_lines_for_patch
from fathom.detection.ml_train import build_model

CONSOLE = Console()

SNR_BUCKETS = [
    (-np.inf, 0.0, "<0"),
    (0.0, 5.0, "0-5"),
    (5.0, 8.0, "5-8"),
    (8.0, 12.0, "8-12"),
    (12.0, 20.0, "12-20"),
    (20.0, np.inf, ">=20"),
]

# Hardcoded adjacency collapse rules per team review 2026-05-25.
MERGED_BUCKETS = [
    (-np.inf, 5.0, "<5"),
    (5.0, 8.0, "5-8"),
    (8.0, 12.0, "8-12"),
    (12.0, np.inf, ">=12"),
]

MIN_N_PER_BUCKET = 10


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


def _collect_per_patch(
    models: list[torch.nn.Module],
    dataset: SyntheticPatchDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-patch (max_mean_confidence, binary_label, peak_snr_db)."""
    stft = dataset.lofar_config.stft
    full_freqs = np.fft.rfftfreq(stft.n_fft, d=1.0 / stft.sample_rate)
    band_mask = (
        (full_freqs >= dataset.lofar_config.freq_min_hz)
        & (full_freqs <= dataset.lofar_config.freq_max_hz)
    )
    gram_freqs = full_freqs[band_mask]
    ps = dataset.patch_config.patch_size
    clip_frame_times: dict[int, np.ndarray] = {}
    confs: list[float] = []
    labels: list[int] = []
    snrs: list[float] = []

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
        masks_stack = np.stack(member_masks, axis=0)
        confs.append(patch_confidence(masks_stack, method="max_mean"))

        truth_lines = extract_truth_lines_for_patch(
            entry.manifest.lines, gram_freqs, frame_times,
            f_start, t_start, ps,
        )
        labels.append(1 if truth_lines else 0)
        if truth_lines:
            snrs.append(float(max(t.peak_snr_db for t in truth_lines)))
        else:
            snrs.append(float("nan"))

    return (
        np.array(confs, dtype=np.float64),
        np.array(labels, dtype=np.int64),
        np.array(snrs, dtype=np.float64),
    )


def _bucket_ece(
    confs: np.ndarray, labels: np.ndarray, snrs: np.ndarray,
    buckets: list[tuple[float, float, str]],
) -> list[dict]:
    """Per-bucket ECE. Negative patches (snr=nan) excluded from all buckets."""
    rows: list[dict] = []
    finite_mask = np.isfinite(snrs)
    for lo, hi, label in buckets:
        mask = finite_mask & (snrs >= lo) & (snrs < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bucket": label, "lower_db": float(lo), "upper_db": float(hi),
                "n_patches": 0, "ece": None, "stable": False,
            })
            continue
        bins = compute_reliability_bins(confs[mask], labels[mask], n_bins=10)
        rows.append({
            "bucket": label, "lower_db": float(lo), "upper_db": float(hi),
            "n_patches": n,
            "ece": float(bins["ece"]),
            "occupied_bins": sum(1 for b in bins["bins"] if b["n_samples"] > 0),
            "stable": n >= MIN_N_PER_BUCKET,
        })
    return rows


def _render_bar_chart(
    rows: list[dict], title: str, out_path: Path,
) -> None:
    """Bar chart of ECE per bucket with n annotated; gray bars for unstable."""
    rows = [r for r in rows if r["n_patches"] > 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [r["bucket"] for r in rows]
    eces = [r["ece"] for r in rows]
    ns = [r["n_patches"] for r in rows]
    stable = [r["stable"] for r in rows]
    colors = ["steelblue" if s else "lightgray" for s in stable]
    x = np.arange(len(labels))
    ax.bar(x, eces, color=colors, edgecolor="black")
    for xi, e, n, s in zip(x, eces, ns, stable):
        annotation = f"n={n}" + ("" if s else " (unstable)")
        ax.text(xi, e + 0.005, annotation, ha="center", fontsize=9)
    ax.axhline(0.05, color="red", lw=1, linestyle="--", alpha=0.6, label="Gate 2 target (0.05)")
    ax.axhline(0.0746, color="orange", lw=1, linestyle="--", alpha=0.6, label="Sprint 5 baseline (0.0746)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Peak SNR bucket (dB)")
    ax.set_ylabel("ECE")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


@click.command()
@click.option(
    "--checkpoints", "checkpoints", multiple=True,
    type=click.Path(exists=True, path_type=Path), required=True,
)
@click.option(
    "--val-data-dir", type=click.Path(exists=True, path_type=Path), required=True,
)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), required=True,
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
        f"[cyan]Val: {len(val_ds)} patches / {len(val_ds._clip_entries)} clips[/cyan]"
    )

    models = [_load_member(cp, unet_base_channels, device_obj) for cp in checkpoints]

    CONSOLE.print("\n[cyan]Running val inference...[/cyan]")
    confs, labels, snrs = _collect_per_patch(models, val_ds, device_obj)
    n_finite_snr = int(np.isfinite(snrs).sum())
    CONSOLE.print(
        f"  {len(confs)} patches, {int(labels.sum())} positive, "
        f"{n_finite_snr} with finite SNR"
    )

    raw_rows = _bucket_ece(confs, labels, snrs, SNR_BUCKETS)
    merged_rows = _bucket_ece(confs, labels, snrs, MERGED_BUCKETS)

    CONSOLE.print("\n[bold cyan]Raw 6-bucket per-SNR calibration[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("bucket"); table.add_column("n", justify="right")
    table.add_column("ECE", justify="right"); table.add_column("stable?", justify="center")
    for r in raw_rows:
        table.add_row(
            r["bucket"], str(r["n_patches"]),
            "—" if r["ece"] is None else f"{r['ece']:.4f}",
            "YES" if r.get("stable") else "no",
        )
    CONSOLE.print(table)

    CONSOLE.print("\n[bold cyan]Merged 4-bucket per-SNR calibration[/bold cyan]")
    CONSOLE.print("[dim](low-SNR merge: <0 + 0-5 -> <5; high-SNR merge: 12-20 + >=20 -> >=12)[/dim]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("bucket"); table.add_column("n", justify="right")
    table.add_column("ECE", justify="right"); table.add_column("stable?", justify="center")
    for r in merged_rows:
        table.add_row(
            r["bucket"], str(r["n_patches"]),
            "—" if r["ece"] is None else f"{r['ece']:.4f}",
            "YES" if r.get("stable") else "no",
        )
    CONSOLE.print(table)

    # Render bar chart of whichever bucketing has more stable buckets
    stable_raw = sum(1 for r in raw_rows if r.get("stable"))
    stable_merged = sum(1 for r in merged_rows if r.get("stable"))
    if stable_merged > stable_raw:
        primary_rows = merged_rows
        chart_title = "Per-SNR-bucket ECE (merged, low/high-SNR collapsed for stability)"
    else:
        primary_rows = raw_rows
        chart_title = "Per-SNR-bucket ECE (raw 6-bucket)"
    _render_bar_chart(primary_rows, chart_title, output_dir / "per_snr_calibration.png")
    CONSOLE.print(f"\n[green]Wrote {output_dir / 'per_snr_calibration.png'}[/green]")

    out_json = {
        "computed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoints": [str(p) for p in checkpoints],
        "val_data_dir": str(val_data_dir),
        "n_patches_total": int(len(confs)),
        "n_patches_positive": int(labels.sum()),
        "n_patches_finite_snr": n_finite_snr,
        "min_n_per_bucket_for_stability": MIN_N_PER_BUCKET,
        "raw_buckets": raw_rows,
        "merged_buckets": merged_rows,
        "primary_for_chart": "merged" if stable_merged > stable_raw else "raw",
    }
    (output_dir / "per_snr_calibration.json").write_text(
        json.dumps(out_json, indent=2)
    )
    CONSOLE.print(f"[green]Wrote {output_dir / 'per_snr_calibration.json'}[/green]")


if __name__ == "__main__":
    main()