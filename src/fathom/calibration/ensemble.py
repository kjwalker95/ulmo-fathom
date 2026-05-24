"""Sprint 6 Cluster C — ensemble scoring functions for calibrated uncertainty.

Platform-layer module. Operates on per-pixel per-member sigmoid mask tensors
of shape (N, H, W) where N is the number of ensemble members. Five scoring
functions + three patch-level aggregation methods.

C5 bimodal-saturation framing: individual U-Net sigmoid masks collapse to
{~0, ~1}; the empty mid-confidence bins between 0.1-0.9 are where graded
confidence lives. Ensemble disagreement creates interior mass through
sigmoid-space averaging (Lakshminarayanan et al. 2017): 3 members at ~0.98
+ 2 members at ~0.02 -> ensemble mean ~0.59, lands in empty bin 5.

Scoring functions (each returns per-pixel scores of shape (H, W)):
  mean_prediction:               ensemble mean. Naive baseline.
  predictive_entropy:            H(mean(p_i)). Total uncertainty.
  mutual_information:            H(mean(p_i)) - mean(H(p_i)). Epistemic.
  member_disagreement_variance:  Var across members. Simple disagreement.
  max_disagreement_patch_score:  scalar - max-pixel Var per patch.

Patch-level aggregation (reduces (N, H, W) to scalar in [0, 1]):
  max_mean:        max pixel of ensemble mean. Sprint 5 C5 baseline.
  mean_max:        mean across members of per-member max.
  peak_freq_band:  max along frequency axis, mean over time.
                   Tuor-defensible operationally (LOFAR line-of-interest reading).

Mask shape convention: (H, W) where H = frequency axis, W = time axis.
This matches src/fathom/detection/ml_eval.py:extract_predicted_lines_mask
which does `freq_idx = int(np.median(rows))` after `np.where(labeled == ...)`.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12  # Numerical floor for log(0) and division by zero


def _check_3d(masks: np.ndarray) -> None:
    if masks.ndim != 3:
        raise ValueError(
            f"masks must be 3D of shape (N, H, W); got shape {masks.shape}"
        )


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy in nats: H(p) = -p log(p) - (1-p) log(1-p).

    Casts to float64 before clamping so EPS actually nudges saturated
    sigmoid outputs away from {0, 1} (in float32, 1.0 - 1e-12 == 1.0
    due to precision loss, leaving log(0) = -inf and 0 * -inf = nan).
    """
    p = np.asarray(p, dtype=np.float64)
    p_safe = np.clip(p, EPS, 1.0 - EPS)
    return -(p_safe * np.log(p_safe) + (1.0 - p_safe) * np.log(1.0 - p_safe))


def mean_prediction(masks: np.ndarray) -> np.ndarray:
    """Element-wise mean of N member sigmoid masks.

    Args:
        masks: shape (N, H, W) — per-member sigmoid mask predictions.

    Returns:
        shape (H, W) — ensemble mean per pixel.
        High = ensemble confident pixel has a line.
        Low  = ensemble confident pixel does NOT have a line.
        Mid  = members disagree about this pixel.
    """
    _check_3d(masks)
    return masks.mean(axis=0)


def predictive_entropy(masks: np.ndarray) -> np.ndarray:
    """Predictive entropy of ensemble mean: H(mean(p_i)).

    Peaks at p=0.5 (maximum uncertainty). Captures TOTAL uncertainty
    (epistemic + aleatoric combined).

    Args:
        masks: shape (N, H, W).

    Returns:
        shape (H, W) — per-pixel entropy in nats.
        High = high total uncertainty (ensemble mean near 0.5).
        Low  = high confidence (ensemble mean near {0, 1}).
    """
    _check_3d(masks)
    p = masks.mean(axis=0)
    return _binary_entropy(p)


def mutual_information(masks: np.ndarray) -> np.ndarray:
    """Mutual information / BALD score for binary classification.

    I = H(mean(p_i)) - mean(H(p_i))

    Captures EPISTEMIC uncertainty — the part of total uncertainty that
    the ensemble disagrees about. Aleatoric uncertainty (inherent noise)
    is captured by mean(H(p_i)); subtracting it leaves what's unknown to
    the ensemble. Smith & Gal 2018; Houlsby et al. 2011 (BALD).

    Args:
        masks: shape (N, H, W).

    Returns:
        shape (H, W) — per-pixel mutual information in nats. Non-negative.
        High = members disagree about this pixel even though individual
               members are confident (epistemic uncertainty).
        Low  = all members agree OR all members individually uncertain.
    """
    _check_3d(masks)
    p_mean = masks.mean(axis=0)
    total = _binary_entropy(p_mean)
    aleatoric = _binary_entropy(masks).mean(axis=0)
    return total - aleatoric


def member_disagreement_variance(masks: np.ndarray) -> np.ndarray:
    """Variance across members at each pixel.

    Simple, interpretable measure of "how much do the N members disagree
    on this pixel?" Population variance (ddof=0, numpy default).

    Args:
        masks: shape (N, H, W).

    Returns:
        shape (H, W) — per-pixel variance. Non-negative.
        High = members disagree.
        Low  = members agree.
    """
    _check_3d(masks)
    return masks.var(axis=0)


def max_disagreement_patch_score(masks: np.ndarray) -> float:
    """Max pixel-level member-disagreement variance over a patch.

    Single scalar per patch. Strong candidate for Cluster D conformal
    nonconformity score: "how uncertain is the ensemble about the most
    contested pixel in this patch?"

    Args:
        masks: shape (N, H, W).

    Returns:
        scalar float — max of member_disagreement_variance over (H, W).
    """
    _check_3d(masks)
    return float(member_disagreement_variance(masks).max())


def patch_confidence(masks: np.ndarray, method: str = "max_mean") -> float:
    """Reduce (N, H, W) member masks to scalar patch confidence in [0, 1].

    Used for reliability-diagram binning and as input to Cluster D's
    conformal nonconformity score (typically as 1 - confidence).

    Args:
        masks: shape (N, H, W) — per-member sigmoid masks.
        method: one of
          "max_mean":
              max pixel of ensemble mean. Sprint 5 C5 baseline. Biases
              toward the hottest pixel — even diffuse ensemble disagreement
              re-saturates here because ONE pixel high in the mean is enough.
          "mean_max":
              mean across members of per-member max. More robust to single-
              pixel outliers; each member contributes its top score and
              ensemble averages.
          "peak_freq_band":
              max along frequency axis (axis=0 of (H, W)), then mean over
              time (axis=1). Tuor-defensible operationally — operators read
              LOFAR grams along the frequency axis (line-of-interest is the
              contiguous vertical structure).

    Returns:
        scalar float in [0, 1].
    """
    _check_3d(masks)

    if method == "max_mean":
        ensemble_mean = masks.mean(axis=0)
        return float(ensemble_mean.max())

    if method == "mean_max":
        per_member_max = masks.max(axis=(1, 2))
        return float(per_member_max.mean())

    if method == "peak_freq_band":
        ensemble_mean = masks.mean(axis=0)
        peak_per_time = ensemble_mean.max(axis=0)
        return float(peak_per_time.mean())

    raise ValueError(
        f"unknown patch_confidence method {method!r}; "
        "expected one of: max_mean, mean_max, peak_freq_band"
    )



def compute_reliability_bins(
    pred_scores: np.ndarray,
    truth_labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Bin predicted patch scores in [0, 1] and compute ECE + per-bin observed rates.

    Standard Guo et al. 2017 reliability binning. Returns the same schema as
    Sprint 5 C5's measure_calibration.py wrote for backward-compatible loading.

    ECE = sum over bins of (n_bin / N_total) * |mean_predicted - observed_rate|.

    Args:
        pred_scores: shape (N,) — patch-level predicted probabilities in [0, 1].
        truth_labels: shape (N,) — patch-level binary labels in {0, 1} or {True, False}.
        n_bins: number of equal-width bins (default 10).

    Returns:
        dict with keys: n_patches, n_positives, n_bins, ece,
        overconfidence_bin_fraction, bins (list of per-bin dicts each with
        bin_idx, lower, upper, n_samples, mean_predicted, observed_positive_rate,
        abs_calibration_gap; mean_predicted/observed/gap are None for empty bins).
    """
    pred = np.asarray(pred_scores, dtype=np.float64).flatten()
    truth = np.asarray(truth_labels, dtype=np.float64).flatten()
    if pred.shape != truth.shape:
        raise ValueError(
            f"pred_scores and truth_labels must have same length; "
            f"got {pred.shape} vs {truth.shape}"
        )
    n_total = len(pred)
    if n_total == 0:
        raise ValueError("empty pred_scores/truth_labels")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    ece = 0.0
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