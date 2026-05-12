"""Tier-1 evaluation harness for the ML line detector (A3 §4 + revision delta).

Converts model outputs → predicted lines (per architecture), matches against
truth-manifest lines via Hungarian assignment with line-IoU cost, aggregates
per-SNR-bucket precision/recall/F1.

Sprint 4 acceptance gate: ≥80% recall at SNR ≥ 8 dB.

Operates at PATCH level. Per-clip line stitching is deferred to Sprint 5+.

Line-IoU (revision delta, locked):
  line_iou = temporal_overlap_ratio × freq_proximity_weight

  temporal_overlap_ratio = |intersection| / |union| of time intervals
  freq_proximity_weight =
    1.0 if |f_pred - f_true| ≤ 2 bins
    0.5 if |f_pred - f_true| ≤ 4 bins
    0.0 otherwise

Match acceptance threshold: line_iou ≥ 0.1 (rejects degenerate Hungarian
assignments where no overlap exists).

Bucketed by truth line's PEAK SNR (not mean — mean is biased low by decay
envelope; peak is what operators care about).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import label as ndimage_label
from scipy.optimize import linear_sum_assignment

from fathom.detection.ml_data import SyntheticPatchDataset
from fathom.grams.lofar import compute_lofar_gram

logger = logging.getLogger(__name__)


SNR_BUCKETS: list[tuple[float, float, str]] = [
    (-float("inf"), 0.0, "<0"),
    (0.0, 5.0, "0-5"),
    (5.0, 8.0, "5-8"),
    (8.0, 12.0, "8-12"),
    (12.0, 20.0, "12-20"),
    (20.0, float("inf"), ">=20"),
]
ACCEPTANCE_BUCKET_THRESHOLDS = (8.0, float("inf"))  # ≥80% recall here is the gate


@dataclass(frozen=True)
class PredictedLine:
    freq_hz: float
    t_start_s: float
    t_end_s: float
    confidence: float


@dataclass(frozen=True)
class TruthLine:
    freq_hz: float
    t_start_s: float
    t_end_s: float
    peak_snr_db: float
    line_id: str


def _temporal_overlap_ratio(
    p_start: float, p_end: float, t_start: float, t_end: float,
) -> float:
    inter = min(p_end, t_end) - max(p_start, t_start)
    if inter <= 0:
        return 0.0
    union = max(p_end, t_end) - min(p_start, t_start)
    return inter / union if union > 0 else 0.0


def _freq_proximity_weight(
    pred_freq_hz: float, truth_freq_hz: float, freq_resolution_hz: float,
) -> float:
    delta_bins = abs(pred_freq_hz - truth_freq_hz) / freq_resolution_hz
    if delta_bins <= 2.0:
        return 1.0
    if delta_bins <= 4.0:
        return 0.5
    return 0.0


def line_iou(
    pred: PredictedLine, truth: TruthLine, freq_resolution_hz: float,
) -> float:
    fpw = _freq_proximity_weight(pred.freq_hz, truth.freq_hz, freq_resolution_hz)
    if fpw == 0.0:
        return 0.0
    tor = _temporal_overlap_ratio(
        pred.t_start_s, pred.t_end_s, truth.t_start_s, truth.t_end_s,
    )
    return fpw * tor


def hungarian_match(
    predictions: list[PredictedLine],
    truths: list[TruthLine],
    freq_resolution_hz: float,
    iou_threshold: float = 0.1,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Optimal one-to-one match maximizing total line-IoU.

    Returns:
      matches: list of (pred_idx, truth_idx, iou) above iou_threshold
      unmatched_pred_idxs: predictions not matched (false positives)
      unmatched_truth_idxs: truths not matched (false negatives)
    """
    n_pred = len(predictions)
    n_truth = len(truths)
    if n_pred == 0:
        return [], [], list(range(n_truth))
    if n_truth == 0:
        return [], list(range(n_pred)), []

    cost = np.ones((n_pred, n_truth))
    for i, pred in enumerate(predictions):
        for j, truth in enumerate(truths):
            cost[i, j] = 1.0 - line_iou(pred, truth, freq_resolution_hz)

    pred_idx, truth_idx = linear_sum_assignment(cost)
    matches: list[tuple[int, int, float]] = []
    matched_pred: set[int] = set()
    matched_truth: set[int] = set()
    for pi, ti in zip(pred_idx, truth_idx):
        iou = 1.0 - cost[pi, ti]
        if iou >= iou_threshold:
            matches.append((int(pi), int(ti), float(iou)))
            matched_pred.add(int(pi))
            matched_truth.add(int(ti))
    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_truth = [j for j in range(n_truth) if j not in matched_truth]
    return matches, unmatched_pred, unmatched_truth


def extract_predicted_lines_heatmap(
    class_prob: float,
    heatmap_probs: np.ndarray,
    patch_freq_axis_hz: np.ndarray,
    patch_t_start_s: float,
    patch_t_end_s: float,
    class_threshold: float = 0.5,
    bin_threshold: float = 0.5,
) -> list[PredictedLine]:
    """ResNet-18 patch-CNN: heatmap-bin peaks above threshold on positive patches.
    Time extent = full patch interval (we don't extract per-frame time bounds here)."""
    if class_prob < class_threshold:
        return []
    out: list[PredictedLine] = []
    for bin_idx in np.where(heatmap_probs > bin_threshold)[0]:
        out.append(PredictedLine(
            freq_hz=float(patch_freq_axis_hz[bin_idx]),
            t_start_s=patch_t_start_s,
            t_end_s=patch_t_end_s,
            confidence=float(heatmap_probs[bin_idx]),
        ))
    return out


def extract_predicted_lines_mask(
    mask_probs: np.ndarray,
    patch_freq_axis_hz: np.ndarray,
    patch_t_start_s: float,
    patch_frame_duration_s: float,
    bin_threshold: float = 0.5,
) -> list[PredictedLine]:
    """U-Net segmentation: connected components in thresholded mask → predicted lines.
    Each component yields (median-freq, time-extent-from-cols, mean-prob)."""
    binary = mask_probs > bin_threshold
    if not binary.any():
        return []
    labeled, n_comp = ndimage_label(binary)
    out: list[PredictedLine] = []
    for comp_id in range(1, n_comp + 1):
        rows, cols = np.where(labeled == comp_id)
        if rows.size == 0:
            continue
        freq_idx = int(np.median(rows))
        t_start = patch_t_start_s + float(cols.min()) * patch_frame_duration_s
        t_end = patch_t_start_s + float(cols.max() + 1) * patch_frame_duration_s
        confidence = float(mask_probs[labeled == comp_id].mean())
        out.append(PredictedLine(
            freq_hz=float(patch_freq_axis_hz[freq_idx]),
            t_start_s=t_start,
            t_end_s=t_end,
            confidence=confidence,
        ))
    return out


def extract_truth_lines_for_patch(
    manifest_lines: list,
    gram_freqs: np.ndarray,
    frame_times: np.ndarray,
    f_start: int,
    t_start: int,
    patch_size: int,
) -> list[TruthLine]:
    """Truth lines intersecting the patch window. Peak SNR per line for bucketing."""
    out: list[TruthLine] = []
    for line in manifest_lines:
        if not line.mask_bin_indices or not line.freq_curve_hz:
            continue
        in_frames: list[int] = []
        in_freqs: list[float] = []
        for k, (frame_idx, _) in enumerate(line.mask_bin_indices):
            if not (t_start <= frame_idx < t_start + patch_size):
                continue
            freq_hz = float(line.freq_curve_hz[k])
            gram_bin = int(np.argmin(np.abs(gram_freqs - freq_hz)))
            if not (f_start <= gram_bin < f_start + patch_size):
                continue
            in_frames.append(frame_idx)
            in_freqs.append(freq_hz)
        if not in_frames:
            continue
        peak_snr = float(max(line.snr_curve_db)) if line.snr_curve_db else 0.0
        out.append(TruthLine(
            freq_hz=float(np.mean(in_freqs)),
            t_start_s=float(frame_times[min(in_frames)]),
            t_end_s=float(frame_times[max(in_frames)] + (frame_times[1] - frame_times[0]) if len(frame_times) > 1 else 0),
            peak_snr_db=peak_snr,
            line_id=line.line_id,
        ))
    return out


def _snr_bucket_label(snr_db: float) -> str:
    for lo, hi, label in SNR_BUCKETS:
        if lo <= snr_db < hi:
            return label
    return SNR_BUCKETS[-1][2]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: SyntheticPatchDataset,
    device: torch.device,
    architecture: str,
    class_threshold: float = 0.5,
    bin_threshold: float = 0.5,
    iou_threshold: float = 0.1,
) -> dict:
    """Run full patch-level Tier-1 evaluation on `dataset`.

    Returns a metrics dict with per-bucket P/R/F1 + overall counts + acceptance-gate verdict.
    """
    model.eval()
    stft = dataset.lofar_config.stft
    full_freqs = np.fft.rfftfreq(stft.n_fft, d=1.0 / stft.sample_rate)
    band_mask = (
        (full_freqs >= dataset.lofar_config.freq_min_hz)
        & (full_freqs <= dataset.lofar_config.freq_max_hz)
    )
    gram_freqs = full_freqs[band_mask]
    freq_resolution_hz = stft.sample_rate / stft.n_fft
    frame_duration_s = stft.hop_length / stft.sample_rate

    # Per-bucket counters
    bucket_tp: dict[str, int] = {label: 0 for _, _, label in SNR_BUCKETS}
    bucket_fn: dict[str, int] = {label: 0 for _, _, label in SNR_BUCKETS}
    total_fp = 0
    total_tp = 0
    total_truth = 0

    # Pre-compute frame_times once per clip (same for all patches in a clip)
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

        # Truth lines for this patch
        truth_lines = extract_truth_lines_for_patch(
            entry.manifest.lines, gram_freqs, frame_times,
            f_start, t_start, ps,
        )
        total_truth += len(truth_lines)

        # Model prediction
        gram = dataset._get_gram(clip_idx, entry)
        patch_np = gram.normalized_power_db[f_start:f_start + ps, t_start:t_start + ps].astype(np.float32)
        patch_t = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

        if architecture == "resnet18":
            class_logits, heatmap_logits = model(patch_t)
            class_prob = float(torch.sigmoid(class_logits).item())
            heatmap_probs = torch.sigmoid(heatmap_logits[0]).cpu().numpy()
            patch_t_start_s = float(frame_times[t_start])
            patch_t_end_s = (
                float(frame_times[t_start + ps - 1] + frame_duration_s)
                if t_start + ps - 1 < len(frame_times)
                else float(frame_times[-1] + frame_duration_s)
            )
            predicted = extract_predicted_lines_heatmap(
                class_prob, heatmap_probs,
                patch_freq_axis_hz=gram_freqs[f_start:f_start + ps],
                patch_t_start_s=patch_t_start_s,
                patch_t_end_s=patch_t_end_s,
                class_threshold=class_threshold,
                bin_threshold=bin_threshold,
            )
        elif architecture == "unet":
            mask_logits = model(patch_t)
            mask_probs = torch.sigmoid(mask_logits[0]).cpu().numpy()
            patch_t_start_s = float(frame_times[t_start])
            predicted = extract_predicted_lines_mask(
                mask_probs,
                patch_freq_axis_hz=gram_freqs[f_start:f_start + ps],
                patch_t_start_s=patch_t_start_s,
                patch_frame_duration_s=frame_duration_s,
                bin_threshold=bin_threshold,
            )
        else:
            raise ValueError(f"unknown architecture {architecture!r}")

        # Hungarian match
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

    # Per-bucket P/R/F1
    buckets_out: dict[str, dict] = {}
    for _, _, label in SNR_BUCKETS:
        tp = bucket_tp[label]
        fn = bucket_fn[label]
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        buckets_out[label] = {
            "tp": tp,
            "fn": fn,
            "n_truth": tp + fn,
            "recall": recall,
        }

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
    overall_recall = total_tp / total_truth if total_truth > 0 else float("nan")
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if overall_precision + overall_recall > 0
        else float("nan")
    )

    # Acceptance gate: ≥80% recall at SNR ≥ 8 dB
    gate_tp = sum(bucket_tp[lab] for _, _, lab in SNR_BUCKETS if lab in ("8-12", "12-20", ">=20"))
    gate_fn = sum(bucket_fn[lab] for _, _, lab in SNR_BUCKETS if lab in ("8-12", "12-20", ">=20"))
    gate_n = gate_tp + gate_fn
    gate_recall = gate_tp / gate_n if gate_n > 0 else float("nan")
    gate_passed = gate_recall >= 0.80 if gate_n > 0 else False

    return {
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
            "passed": gate_passed,
        },
    }


def print_eval_summary(metrics: dict, console=None) -> None:
    """Pretty-print eval results to a rich console (or stdout)."""
    if console is None:
        from rich.console import Console as _Console
        console = _Console()

    console.print("\n[cyan]── Tier-1 Evaluation (patch-level) ──[/cyan]")
    console.print(f"  Overall: P={metrics['overall']['precision']:.3f}  "
                  f"R={metrics['overall']['recall']:.3f}  "
                  f"F1={metrics['overall']['f1']:.3f}  "
                  f"(TP={metrics['overall']['tp']}, FP={metrics['overall']['fp']}, "
                  f"truth={metrics['overall']['total_truth']})")
    console.print("\n  Per-SNR-bucket recall (truth peak SNR):")
    console.print(f"    {'bucket':<10} {'n_truth':>8} {'tp':>5} {'fn':>5} {'recall':>8}")
    for _, _, label in SNR_BUCKETS:
        b = metrics["buckets"][label]
        r_str = f"{b['recall']:.3f}" if b["n_truth"] > 0 else "  n/a"
        console.print(f"    {label:<10} {b['n_truth']:>8} {b['tp']:>5} {b['fn']:>5} {r_str:>8}")

    gate = metrics["acceptance_gate"]
    verdict = "[green]PASS[/green]" if gate["passed"] else "[red]FAIL[/red]"
    actual_str = (
        f"{gate['actual_recall']:.3f}" if gate["n_truth"] > 0 else "n/a (no truth lines)"
    )
    console.print(
        f"\n  Acceptance gate: recall @ SNR >= {gate['snr_threshold_db']:.0f} dB "
        f">= {gate['target_recall']:.2f}  →  actual={actual_str}  [{verdict}]"
    )