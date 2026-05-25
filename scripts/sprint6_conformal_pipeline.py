"""Sprint 6 Cluster D.2-D.6 — full conformal pipeline end to end.

Runs:
  D.2  Inference on Tier-3 reserve calibration set (5 ensemble members)
       -> per-patch (confidence, label) pairs.
  D.3  Fit ConformalCalibrator at union of {0.05, 0.10, 0.20} (committed)
       and {0.01, 0.02, ..., 0.50} (for coverage curve). Persist JSON.
  D.4  Inference on Tier-2 val set -> coverage curves per-class.
  D.5  Prediction-set-size analysis vs SNR bucket on val.
  D.6  360-beam per-beam FAR projection at the 3 committed alpha levels.

Outputs under <output-dir>/:
  calibrator.json             (D.3 persisted state)
  coverage_curve.png          (D.4 per-class curves)
  coverage_curve.json         (D.4 raw data, 50 alpha levels)
  prediction_set_sizes.json   (D.5)
  set_size_vs_snr.png         (D.5 plot)
  far_projection.md           (D.6 markdown table for commit message)

Per the D.0 pre-flight + team review: Tier-3 reserve has only 3 viable
vessels (~180 patches); alpha=0.05 reported in coverage curves but the
D.6 FAR commitment is limited to {0.10, 0.20} per the negative-class
finite-sample bound rationale.
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

from fathom.calibration.conformal import (
    ConformalCalibrator,
    empirical_coverage,
)
from fathom.calibration.ensemble import patch_confidence
from fathom.detection.ml_data import (
    PatchExtractionConfig,
    SyntheticPatchDataset,
    default_lofar_config,
)
from fathom.detection.ml_eval import extract_truth_lines_for_patch
from fathom.detection.ml_train import build_model

CONSOLE = Console()

COMMITTED_ALPHAS = (0.05, 0.10, 0.20)
CURVE_ALPHAS = tuple(round(0.01 * i, 2) for i in range(1, 51))
ALL_ALPHAS = tuple(sorted(set(COMMITTED_ALPHAS) | set(CURVE_ALPHAS)))
FAR_COMMITTED_ALPHAS = (0.10, 0.20)


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


def _collect_confidences_and_labels(
    models: list[torch.nn.Module],
    dataset: SyntheticPatchDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-patch (max_mean confidence, binary label, peak_snr_db).

    peak_snr_db is NaN for negative patches; max(peak_snr_db across truth
    lines in patch) for positives.
    """
    stft = dataset.lofar_config.stft
    full_freqs = np.fft.rfftfreq(stft.n_fft, d=1.0 / stft.sample_rate)
    band_mask = (
        (full_freqs >= dataset.lofar_config.freq_min_hz)
        & (full_freqs <= dataset.lofar_config.freq_max_hz)
    )
    gram_freqs = full_freqs[band_mask]

    ps = dataset.patch_config.patch_size
    clip_frame_times: dict[int, np.ndarray] = {}
    confidences: list[float] = []
    labels: list[int] = []
    peak_snrs: list[float] = []

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

        conf = patch_confidence(masks_stack, method="max_mean")
        confidences.append(conf)

        truth_lines = extract_truth_lines_for_patch(
            entry.manifest.lines, gram_freqs, frame_times,
            f_start, t_start, ps,
        )
        labels.append(1 if truth_lines else 0)
        if truth_lines:
            peak_snrs.append(float(max(t.peak_snr_db for t in truth_lines)))
        else:
            peak_snrs.append(float("nan"))

    return (
        np.array(confidences, dtype=np.float64),
        np.array(labels, dtype=np.int64),
        np.array(peak_snrs, dtype=np.float64),
    )


def _patches_per_hour(lofar_config, patch_size: int, stride: int) -> float:
    """Derive patches-per-hour-per-beam from the live LOFAR config."""
    stft = lofar_config.stft
    frame_duration_s = stft.hop_length / stft.sample_rate
    stride_duration_s = stride * frame_duration_s
    return 3600.0 / stride_duration_s


@click.command()
@click.option(
    "--checkpoints", "checkpoints", multiple=True,
    type=click.Path(exists=True, path_type=Path), required=True,
)
@click.option(
    "--cal-data-dir", type=click.Path(exists=True, path_type=Path), required=True,
    help="Tier-3 reserve calibration dataset",
)
@click.option(
    "--val-data-dir", type=click.Path(exists=True, path_type=Path), required=True,
    help="Tier-2 val evaluation dataset",
)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), required=True,
)
@click.option("--device", type=str, default="auto")
@click.option("--unet-base-channels", type=int, default=64)
def main(
    checkpoints: tuple[Path, ...],
    cal_data_dir: Path,
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
    CONSOLE.print(f"[cyan]Cal: {cal_data_dir}[/cyan]")
    CONSOLE.print(f"[cyan]Val: {val_data_dir}  device={device_obj}[/cyan]")

    lofar_cfg = default_lofar_config()
    patch_cfg = PatchExtractionConfig(
        patch_size=256, stride=128, target_mode="mask",
    )

    cal_paths = sorted(Path(cal_data_dir).glob("*.wav"))
    val_paths = sorted(Path(val_data_dir).glob("*.wav"))
    cal_ds = SyntheticPatchDataset(
        clip_paths=cal_paths, lofar_config=lofar_cfg, patch_config=patch_cfg,
    )
    val_ds = SyntheticPatchDataset(
        clip_paths=val_paths, lofar_config=lofar_cfg, patch_config=patch_cfg,
    )
    CONSOLE.print(
        f"[cyan]Cal: {len(cal_ds)} patches / {len(cal_ds._clip_entries)} clips[/cyan]"
    )
    CONSOLE.print(
        f"[cyan]Val: {len(val_ds)} patches / {len(val_ds._clip_entries)} clips[/cyan]"
    )

    models = [_load_member(cp, unet_base_channels, device_obj) for cp in checkpoints]

    CONSOLE.print("\n[cyan]D.2 inference on calibration set...[/cyan]")
    cal_confs, cal_labels, _cal_snrs = _collect_confidences_and_labels(
        models, cal_ds, device_obj,
    )
    CONSOLE.print(
        f"  cal: {len(cal_confs)} patches, "
        f"{int(cal_labels.sum())} positive ({100*cal_labels.mean():.1f}%)"
    )

    CONSOLE.print("\n[cyan]D.3 fit ConformalCalibrator...[/cyan]")
    calibrator = ConformalCalibrator.fit(
        cal_confs, cal_labels, alpha_levels=ALL_ALPHAS,
    )
    calibrator.save_json(output_dir / "calibrator.json")
    CONSOLE.print(
        f"  n_cal_pos={calibrator.n_cal_positives}  "
        f"n_cal_neg={calibrator.n_cal_negatives}"
    )
    CONSOLE.print(
        f"  bound_pos={calibrator.finite_sample_bound_positive():.4f}  "
        f"bound_neg={calibrator.finite_sample_bound_negative():.4f}"
    )

    committed_table = Table(show_header=True, header_style="bold magenta")
    committed_table.add_column("alpha")
    committed_table.add_column("q_positive", justify="right")
    committed_table.add_column("q_negative", justify="right")
    for a in COMMITTED_ALPHAS:
        committed_table.add_row(
            f"{a:.2f}",
            f"{calibrator.q_positive[a]:.4f}",
            f"{calibrator.q_negative[a]:.4f}",
        )
    CONSOLE.print(committed_table)
    CONSOLE.print(f"  -> {output_dir / 'calibrator.json'}")

    CONSOLE.print("\n[cyan]D.4 inference on val + coverage curves...[/cyan]")
    val_confs, val_labels, val_snrs = _collect_confidences_and_labels(
        models, val_ds, device_obj,
    )
    CONSOLE.print(
        f"  val: {len(val_confs)} patches, "
        f"{int(val_labels.sum())} positive ({100*val_labels.mean():.1f}%)"
    )

    coverage_rows: list[dict] = []
    for a in ALL_ALPHAS:
        cov = empirical_coverage(calibrator, val_confs, val_labels, a)
        coverage_rows.append(cov)

    # Coverage curve plot (per-class)
    fig, ax = plt.subplots(figsize=(8, 6))
    alphas = np.array([r["alpha"] for r in coverage_rows])
    nominal = 1.0 - alphas
    pos_cov = np.array([r["empirical_coverage_positive"] for r in coverage_rows])
    neg_cov = np.array([r["empirical_coverage_negative"] for r in coverage_rows])
    bp = calibrator.finite_sample_bound_positive()
    bn = calibrator.finite_sample_bound_negative()
    ax.plot(nominal, nominal, "k--", lw=1, alpha=0.5, label="nominal (y=x)")
    ax.plot(nominal, pos_cov, "o-", color="firebrick", label="positive class empirical")
    ax.fill_between(
        nominal, nominal - bp, nominal + bp,
        color="firebrick", alpha=0.15, label=f"positive bound (+/-{bp:.3f})",
    )
    ax.plot(nominal, neg_cov, "s-", color="steelblue", label="negative class empirical")
    ax.fill_between(
        nominal, nominal - bn, nominal + bn,
        color="steelblue", alpha=0.15, label=f"negative bound (+/-{bn:.3f})",
    )
    for a in COMMITTED_ALPHAS:
        ax.axvline(1 - a, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel("Nominal coverage (1 - alpha)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(
        f"Coverage curves - cal n_pos={calibrator.n_cal_positives} "
        f"n_neg={calibrator.n_cal_negatives}"
    )
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / "coverage_curve.png"), dpi=100)
    plt.close(fig)
    CONSOLE.print(f"  -> {output_dir / 'coverage_curve.png'}")

    (output_dir / "coverage_curve.json").write_text(json.dumps({
        "n_cal": calibrator.n_cal,
        "n_cal_positives": calibrator.n_cal_positives,
        "n_cal_negatives": calibrator.n_cal_negatives,
        "finite_sample_bound_positive": bp,
        "finite_sample_bound_negative": bn,
        "rows": coverage_rows,
    }, indent=2))

    committed_cov = [r for r in coverage_rows if r["alpha"] in COMMITTED_ALPHAS]
    cov_table = Table(show_header=True, header_style="bold magenta")
    cov_table.add_column("alpha")
    cov_table.add_column("pos coverage", justify="right")
    cov_table.add_column("neg coverage", justify="right")
    cov_table.add_column("pos in bound?", justify="center")
    cov_table.add_column("neg in bound?", justify="center")
    for r in committed_cov:
        pos_in = abs(r["empirical_coverage_positive"] - r["nominal_coverage"]) <= bp
        neg_in = abs(r["empirical_coverage_negative"] - r["nominal_coverage"]) <= bn
        cov_table.add_row(
            f"{r['alpha']:.2f}",
            f"{r['empirical_coverage_positive']:.3f}",
            f"{r['empirical_coverage_negative']:.3f}",
            "PASS" if pos_in else "MISS",
            "PASS" if neg_in else "MISS",
        )
    CONSOLE.print(cov_table)

    CONSOLE.print("\n[cyan]D.5 prediction-set size analysis vs SNR...[/cyan]")
    snr_buckets = [
        (-np.inf, 0.0, "<0"),
        (0.0, 5.0, "0-5"),
        (5.0, 8.0, "5-8"),
        (8.0, 12.0, "8-12"),
        (12.0, 20.0, "12-20"),
        (20.0, np.inf, ">=20"),
    ]
    set_size_by_bucket: dict[str, dict] = {}
    for lo, hi, label in snr_buckets:
        mask = (val_snrs >= lo) & (val_snrs < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        sizes: list[int] = []
        for c in val_confs[mask]:
            ps = calibrator.predict_set(float(c), alpha=0.10)
            sizes.append(ps.set_size)
        set_size_by_bucket[label] = {
            "n_patches": n,
            "mean_set_size_alpha010": float(np.mean(sizes)),
            "fraction_singleton": float(np.mean([s == 1 for s in sizes])),
            "fraction_uncertain": float(np.mean([s == 2 for s in sizes])),
            "fraction_empty": float(np.mean([s == 0 for s in sizes])),
        }
    (output_dir / "prediction_set_sizes.json").write_text(
        json.dumps(set_size_by_bucket, indent=2)
    )

    # Set-size vs SNR plot at alpha=0.10
    labels_in_order = [
        lab for _, _, lab in snr_buckets if lab in set_size_by_bucket
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels_in_order))
    singletons = [set_size_by_bucket[lab]["fraction_singleton"] for lab in labels_in_order]
    uncertain = [set_size_by_bucket[lab]["fraction_uncertain"] for lab in labels_in_order]
    empty = [set_size_by_bucket[lab]["fraction_empty"] for lab in labels_in_order]
    ax.bar(x, singletons, label="singleton", color="seagreen")
    ax.bar(x, uncertain, bottom=singletons, label="uncertain (both)", color="goldenrod")
    bottom2 = [s + u for s, u in zip(singletons, uncertain)]
    ax.bar(x, empty, bottom=bottom2, label="empty", color="firebrick")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_in_order)
    ax.set_xlabel("Peak SNR bucket (dB)")
    ax.set_ylabel("Fraction of patches")
    ax.set_title("Prediction set size composition vs SNR (alpha=0.10)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(str(output_dir / "set_size_vs_snr.png"), dpi=100)
    plt.close(fig)
    CONSOLE.print(f"  -> {output_dir / 'set_size_vs_snr.png'}")

    CONSOLE.print("\n[cyan]D.6 360-beam FAR projection...[/cyan]")
    pph = _patches_per_hour(
        lofar_cfg, patch_cfg.patch_size, patch_cfg.stride,
    )
    CONSOLE.print(f"  patches/hour/beam (derived from live config): {pph:.1f}")
    neg_mask = (val_labels == 0)
    n_val_neg = int(neg_mask.sum())
    far_rows: list[dict] = []
    for a in FAR_COMMITTED_ALPHAS:
        fp_count = 0
        for c in val_confs[neg_mask]:
            ps = calibrator.predict_set(float(c), alpha=a)
            if ps.verdict == "detected":
                fp_count += 1
        p_fp = fp_count / n_val_neg if n_val_neg > 0 else float("nan")
        far_per_beam = p_fp * pph
        aggregate_180 = far_per_beam * 180
        far_rows.append({
            "alpha": a,
            "n_val_negatives": n_val_neg,
            "false_positive_rate": p_fp,
            "per_beam_far_per_hour": far_per_beam,
            "aggregate_180_beam_alerts_per_hour": aggregate_180,
            "patches_per_hour_per_beam": pph,
        })

    far_table = Table(show_header=True, header_style="bold magenta")
    far_table.add_column("alpha")
    far_table.add_column("p_FP", justify="right")
    far_table.add_column("per-beam FAR/hr", justify="right")
    far_table.add_column("180-beam alerts/hr", justify="right")
    far_table.add_column("<0.05/hr per beam?", justify="center")
    for r in far_rows:
        far_table.add_row(
            f"{r['alpha']:.2f}",
            f"{r['false_positive_rate']:.4f}",
            f"{r['per_beam_far_per_hour']:.4f}",
            f"{r['aggregate_180_beam_alerts_per_hour']:.2f}",
            "PASS" if r["per_beam_far_per_hour"] < 0.05 else "MISS",
        )
    CONSOLE.print(far_table)

    md_lines = [
        "# Sprint 6 D.6 - 360-beam FAR projection",
        "",
        f"- Per the C.4 winner: max_mean confidence + ensemble of 5 U-Nets",
        f"- Calibration set: Tier-3 reserve (n_cal={calibrator.n_cal}, "
        f"pos={calibrator.n_cal_positives}, neg={calibrator.n_cal_negatives})",
        f"- Per-class bounds: positive {bp:.4f}, negative {bn:.4f}",
        f"- Val evaluation set: {len(val_confs)} patches, "
        f"{n_val_neg} negative",
        f"- Patches/hour/beam (derived from default_lofar_config): {pph:.1f}",
        f"- alpha=0.05 fit + plotted but NOT committed for FAR per the D.0 "
        f"thin-negative-class triage; only alpha in {{0.10, 0.20}} committed.",
        "",
        "| alpha | p_FP | per-beam FAR/hr | 180-beam alerts/hr | <0.05/hr? |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for r in far_rows:
        verdict = "PASS" if r["per_beam_far_per_hour"] < 0.05 else "MISS"
        md_lines.append(
            f"| {r['alpha']:.2f} | {r['false_positive_rate']:.4f} | "
            f"{r['per_beam_far_per_hour']:.4f} | "
            f"{r['aggregate_180_beam_alerts_per_hour']:.2f} | {verdict} |"
        )
    md_lines.append("")
    md_lines.append("Watch Supervisor tolerance: 5-10 alerts/hr aggregate.")
    (output_dir / "far_projection.md").write_text("\n".join(md_lines) + "\n")
    CONSOLE.print(f"  -> {output_dir / 'far_projection.md'}")

    CONSOLE.print("\n[green]D.2-D.6 complete.[/green]")


if __name__ == "__main__":
    main()