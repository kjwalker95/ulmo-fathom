"""Sprint 6 Cluster D — split-conformal prediction wrapper for binary patch classification.

Per Vovk et al. 2005 / Angelopoulos & Bates 2023 Theorem 2.2. The (n+1)(1-alpha)/n
corrected quantile gives the exact finite-sample coverage guarantee at >= 1 - alpha
for new exchangeable samples. The plain np.quantile(scores, 1 - alpha) yields only
ASYMPTOTIC coverage; the (n+1) correction is what makes split-conformal
distribution-free.

Binary patch classification (line present / not present):
  Calibration set: (confidence, label) pairs. Confidence is patch-level from
    fathom.calibration.ensemble.patch_confidence (Cluster C.4 winner:
    mean_prediction + max_mean aggregation).
  Nonconformity scores per hypothesized class:
    s_pos(x) = 1 - confidence(x)   lower = better fit for positive hypothesis
    s_neg(x) = confidence(x)        lower = better fit for negative hypothesis
  Per-class quantile thresholds at alpha:
    q_pos[alpha] = corrected (n_pos+1)(1-alpha)/n_pos-quantile of {s_pos(x_i) : y_i = 1}
    q_neg[alpha] = corrected (n_neg+1)(1-alpha)/n_neg-quantile of {s_neg(x_i) : y_i = 0}
  Prediction set for new patch x_new:
    include positive if s_pos(x_new) <= q_pos[alpha]
    include negative if s_neg(x_new) <= q_neg[alpha]
  Verdict mapping:
    {positive}                       -> "detected"
    {negative}                       -> "not_detected"
    {positive, negative}             -> "uncertain"  (analyst review)
    {} (empty)                       -> "empty"      (should be near-zero w/ proper cal)

Finite-sample bounds are PER-CLASS:
  bound_positive = 1 / sqrt(n_cal_positives)
  bound_negative = 1 / sqrt(n_cal_negatives)
At Tier-2's ~78% positive injection rate, negative-class bound is typically 2x
looser than positive-class. Report both honestly; if downstream coverage on val
shows asymmetric deviation (one class in-bound, other out), the explanation is
distribution shift between calibration and evaluation vessels - NOT a conformal
implementation bug.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConformalPredictionSet:
    """One conformal prediction set verdict for a single patch."""
    alpha: float
    confidence: float
    includes_positive: bool
    includes_negative: bool

    @property
    def verdict(self) -> str:
        """Operational verdict: detected / not_detected / uncertain / empty."""
        if self.includes_positive and not self.includes_negative:
            return "detected"
        if self.includes_negative and not self.includes_positive:
            return "not_detected"
        if self.includes_positive and self.includes_negative:
            return "uncertain"
        return "empty"

    @property
    def set_size(self) -> int:
        return int(self.includes_positive) + int(self.includes_negative)


@dataclass(frozen=True)
class ConformalCalibrator:
    """Split-conformal calibrator for binary patch classification.

    Fit on calibration (confidence, label) pairs; apply via predict_set
    to new patches at a chosen alpha level. The (n+1)(1-alpha)/n
    corrected quantile gives exact finite-sample coverage at >= 1 - alpha.
    """
    alpha_levels: tuple[float, ...]
    q_positive: dict[float, float]
    q_negative: dict[float, float]
    n_cal: int
    n_cal_positives: int
    n_cal_negatives: int

    @classmethod
    def fit(
        cls,
        confidences: np.ndarray,
        labels: np.ndarray,
        alpha_levels: tuple[float, ...] = (0.05, 0.10, 0.20),
    ) -> "ConformalCalibrator":
        """Fit per-class quantile thresholds via split conformal.

        Args:
            confidences: shape (N,) - patch-level confidences in [0, 1].
            labels: shape (N,) - binary labels {0, 1}.
            alpha_levels: tuple of significance levels to fit.

        Returns:
            ConformalCalibrator with fitted per-class quantile dicts.
        """
        confidences = np.asarray(confidences, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if confidences.shape != labels.shape:
            raise ValueError(
                f"confidences and labels must have same shape; "
                f"got {confidences.shape} vs {labels.shape}"
            )
        if confidences.ndim != 1:
            raise ValueError(
                f"confidences must be 1D; got shape {confidences.shape}"
            )

        pos_mask = labels.astype(bool)
        scores_pos = 1.0 - confidences[pos_mask]
        scores_neg = confidences[~pos_mask]

        if len(scores_pos) == 0:
            raise ValueError("no positive calibration points; can't fit positive quantile")
        if len(scores_neg) == 0:
            raise ValueError("no negative calibration points; can't fit negative quantile")

        q_pos = {a: _corrected_quantile(scores_pos, a) for a in alpha_levels}
        q_neg = {a: _corrected_quantile(scores_neg, a) for a in alpha_levels}

        return cls(
            alpha_levels=tuple(alpha_levels),
            q_positive=q_pos,
            q_negative=q_neg,
            n_cal=int(len(confidences)),
            n_cal_positives=int(pos_mask.sum()),
            n_cal_negatives=int((~pos_mask).sum()),
        )

    def predict_set(
        self, confidence: float, alpha: float,
    ) -> ConformalPredictionSet:
        """Return prediction set verdict for a new patch.

        Args:
            confidence: patch-level confidence in [0, 1].
            alpha: significance level; must be in self.alpha_levels.
        """
        if alpha not in self.q_positive:
            raise ValueError(
                f"alpha={alpha} not in fitted levels {sorted(self.alpha_levels)}"
            )
        include_pos = (1.0 - confidence) <= self.q_positive[alpha]
        include_neg = confidence <= self.q_negative[alpha]
        return ConformalPredictionSet(
            alpha=alpha,
            confidence=float(confidence),
            includes_positive=bool(include_pos),
            includes_negative=bool(include_neg),
        )

    def finite_sample_bound_positive(self) -> float:
        """1 / sqrt(n_cal_positives). Per-class envelope on positive coverage."""
        return float(1.0 / np.sqrt(self.n_cal_positives))

    def finite_sample_bound_negative(self) -> float:
        """1 / sqrt(n_cal_negatives). Per-class envelope on negative coverage."""
        return float(1.0 / np.sqrt(self.n_cal_negatives))

    def to_dict(self) -> dict:
        """JSON-serializable state."""
        return {
            "alpha_levels": list(self.alpha_levels),
            "q_positive": {str(k): v for k, v in self.q_positive.items()},
            "q_negative": {str(k): v for k, v in self.q_negative.items()},
            "n_cal": self.n_cal,
            "n_cal_positives": self.n_cal_positives,
            "n_cal_negatives": self.n_cal_negatives,
            "finite_sample_bound_positive": self.finite_sample_bound_positive(),
            "finite_sample_bound_negative": self.finite_sample_bound_negative(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalCalibrator":
        """Restore from a to_dict() payload."""
        return cls(
            alpha_levels=tuple(d["alpha_levels"]),
            q_positive={float(k): float(v) for k, v in d["q_positive"].items()},
            q_negative={float(k): float(v) for k, v in d["q_negative"].items()},
            n_cal=int(d["n_cal"]),
            n_cal_positives=int(d["n_cal_positives"]),
            n_cal_negatives=int(d["n_cal_negatives"]),
        )

    def save_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path: Path) -> "ConformalCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _corrected_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (n+1)(1-alpha)/n quantile.

    Vovk et al. 2005; Angelopoulos & Bates 2023 Theorem 2.2. The plain
    np.quantile(scores, 1 - alpha) gives only asymptotic coverage; the
    (n+1) correction enables the distribution-free finite-sample
    coverage guarantee. Clipped to 1.0 for small n / high alpha cases
    where the level exceeds 1.0.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        raise ValueError("empty scores; cannot compute quantile")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level))


def empirical_coverage(
    calibrator: ConformalCalibrator,
    confidences: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> dict:
    """Coverage analysis on a held-out evaluation set.

    Reports per-class coverage separately (positive_coverage, negative_coverage)
    since the per-class quantile thresholds are fit independently and each has
    its own finite-sample bound.

    Args:
        calibrator: fitted ConformalCalibrator.
        confidences: shape (N,) - eval-set patch confidences in [0, 1].
        labels: shape (N,) - eval-set binary labels {0, 1}.
        alpha: significance level (must be in calibrator.alpha_levels).

    Returns:
        dict with empirical coverage (overall + per-class), per-class bounds,
        and prediction-set-size composition (singleton / uncertain / empty
        fractions).
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if confidences.shape != labels.shape:
        raise ValueError(
            f"confidences and labels must have same shape; "
            f"got {confidences.shape} vs {labels.shape}"
        )
    n = len(confidences)
    pos_mask = labels.astype(bool)
    n_pos = int(pos_mask.sum())
    n_neg = int((~pos_mask).sum())

    in_set_pos = 0
    in_set_neg = 0
    set_sizes: list[int] = []
    for c, y in zip(confidences, labels):
        ps = calibrator.predict_set(float(c), alpha)
        if y == 1 and ps.includes_positive:
            in_set_pos += 1
        if y == 0 and ps.includes_negative:
            in_set_neg += 1
        set_sizes.append(ps.set_size)

    return {
        "alpha": alpha,
        "n": n,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "nominal_coverage": 1.0 - alpha,
        "empirical_coverage_overall": (in_set_pos + in_set_neg) / n,
        "empirical_coverage_positive": (
            in_set_pos / n_pos if n_pos > 0 else float("nan")
        ),
        "empirical_coverage_negative": (
            in_set_neg / n_neg if n_neg > 0 else float("nan")
        ),
        "finite_sample_bound_positive": calibrator.finite_sample_bound_positive(),
        "finite_sample_bound_negative": calibrator.finite_sample_bound_negative(),
        "mean_set_size": float(np.mean(set_sizes)),
        "fraction_singleton": float(np.mean([s == 1 for s in set_sizes])),
        "fraction_uncertain": float(np.mean([s == 2 for s in set_sizes])),
        "fraction_empty": float(np.mean([s == 0 for s in set_sizes])),
    }