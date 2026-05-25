"""Tests for fathom.calibration.conformal (Sprint 6 Cluster D)."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fathom.calibration.conformal import (
    ConformalCalibrator,
    ConformalPredictionSet,
    _corrected_quantile,
    empirical_coverage,
)


class TestCorrectedQuantile:
    def test_differs_from_plain_at_small_n(self):
        # At small n, the (n+1)(1-alpha)/n correction shifts the level.
        # For n=10, alpha=0.10: (11)(0.9)/10 = 9.9 -> ceil to 10 -> 10/10 = 1.0
        # -> clip to 1.0 -> returns max(scores). Plain np.quantile at 0.90
        # gives the 9th-decile interpolation, which is strictly less than max.
        rng = np.random.default_rng(20260524)
        scores = rng.uniform(size=10)
        plain = float(np.quantile(scores, 0.90))
        corrected = _corrected_quantile(scores, alpha=0.10)
        assert corrected == float(scores.max())
        assert corrected > plain

    def test_clip_to_one(self):
        # When (n+1)(1-alpha)/n > 1.0, level clips to 1.0 -> max.
        scores = np.array([0.1, 0.5, 0.9])
        # n=3, alpha=0.05: (4)(0.95)/3 = 1.267 -> clip 1.0 -> max
        assert _corrected_quantile(scores, alpha=0.05) == 0.9

    def test_asymptotic_agreement_at_large_n(self):
        rng = np.random.default_rng(20260524)
        scores = rng.uniform(size=10000)
        plain = float(np.quantile(scores, 0.90))
        corrected = _corrected_quantile(scores, alpha=0.10)
        assert abs(corrected - plain) < 0.01

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty scores"):
            _corrected_quantile(np.array([]), alpha=0.10)


class TestConformalCalibratorFit:
    def test_fit_returns_calibrator(self):
        rng = np.random.default_rng(20260524)
        confs = rng.uniform(size=100)
        labels = (confs > 0.5).astype(np.int64)
        cal = ConformalCalibrator.fit(confs, labels, alpha_levels=(0.10,))
        assert isinstance(cal, ConformalCalibrator)
        assert cal.n_cal == 100
        assert 0.10 in cal.q_positive
        assert 0.10 in cal.q_negative

    def test_per_class_counts(self):
        confs = np.array([0.1, 0.2, 0.8, 0.9, 0.95])
        labels = np.array([0, 0, 1, 1, 1])
        cal = ConformalCalibrator.fit(confs, labels)
        assert cal.n_cal_positives == 3
        assert cal.n_cal_negatives == 2

    def test_no_positives_raises(self):
        with pytest.raises(ValueError, match="no positive"):
            ConformalCalibrator.fit(
                np.array([0.1, 0.2]), np.array([0, 0]),
            )

    def test_no_negatives_raises(self):
        with pytest.raises(ValueError, match="no negative"):
            ConformalCalibrator.fit(
                np.array([0.8, 0.9]), np.array([1, 1]),
            )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            ConformalCalibrator.fit(
                np.array([0.5]), np.array([1, 0]),
            )


class TestConformalPredictionSet:
    def test_verdict_detected(self):
        ps = ConformalPredictionSet(
            alpha=0.1, confidence=0.95,
            includes_positive=True, includes_negative=False,
        )
        assert ps.verdict == "detected"
        assert ps.set_size == 1

    def test_verdict_not_detected(self):
        ps = ConformalPredictionSet(
            alpha=0.1, confidence=0.05,
            includes_positive=False, includes_negative=True,
        )
        assert ps.verdict == "not_detected"
        assert ps.set_size == 1

    def test_verdict_uncertain(self):
        ps = ConformalPredictionSet(
            alpha=0.1, confidence=0.5,
            includes_positive=True, includes_negative=True,
        )
        assert ps.verdict == "uncertain"
        assert ps.set_size == 2

    def test_verdict_empty(self):
        # Shouldn't happen w/ proper calibration but the case is defined.
        ps = ConformalPredictionSet(
            alpha=0.1, confidence=0.5,
            includes_positive=False, includes_negative=False,
        )
        assert ps.verdict == "empty"
        assert ps.set_size == 0


class TestFiniteSampleBounds:
    def test_per_class_bounds_use_per_class_counts(self):
        # 90% positive, 10% negative -> bound_neg >> bound_pos.
        rng = np.random.default_rng(20260524)
        labels = np.concatenate([np.ones(90), np.zeros(10)]).astype(np.int64)
        confs = rng.uniform(size=100)
        cal = ConformalCalibrator.fit(confs, labels)
        assert cal.n_cal_positives == 90
        assert cal.n_cal_negatives == 10
        bp = cal.finite_sample_bound_positive()
        bn = cal.finite_sample_bound_negative()
        assert bp == pytest.approx(1.0 / np.sqrt(90), abs=1e-9)
        assert bn == pytest.approx(1.0 / np.sqrt(10), abs=1e-9)
        assert bn > bp


class TestJsonRoundTrip:
    def test_save_load(self):
        rng = np.random.default_rng(20260524)
        confs = rng.uniform(size=100)
        labels = (confs > 0.5).astype(np.int64)
        cal = ConformalCalibrator.fit(confs, labels)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cal.json"
            cal.save_json(p)
            cal2 = ConformalCalibrator.load_json(p)
        assert cal2.alpha_levels == cal.alpha_levels
        assert cal2.n_cal == cal.n_cal
        assert cal2.n_cal_positives == cal.n_cal_positives
        assert cal2.n_cal_negatives == cal.n_cal_negatives
        for alpha in cal.alpha_levels:
            assert cal2.q_positive[alpha] == pytest.approx(cal.q_positive[alpha])
            assert cal2.q_negative[alpha] == pytest.approx(cal.q_negative[alpha])


class TestCoverageSimulation:
    """Load-bearing acceptance test: per-class coverage tracks nominal alpha
    within per-class finite-sample bound on synthetic exchangeable data."""

    def _generate_exchangeable(self, n: int, seed: int, pos_rate: float = 0.78):
        """Synthetic (confidence, label) pairs from a known calibrated DGP."""
        rng = np.random.default_rng(seed)
        labels = (rng.uniform(size=n) < pos_rate).astype(np.int64)
        # Calibrated DGP: positives draw from beta(8, 2) (high-skewed);
        # negatives from beta(2, 8) (low-skewed). Most mass resolves correctly
        # but some confusion exists at the middle.
        confs = np.where(
            labels == 1,
            rng.beta(8.0, 2.0, size=n),
            rng.beta(2.0, 8.0, size=n),
        )
        return confs, labels

    def test_per_class_coverage_within_bounds_single_trial(self):
        cal_confs, cal_labels = self._generate_exchangeable(n=500, seed=1)
        eval_confs, eval_labels = self._generate_exchangeable(n=500, seed=2)
        cal = ConformalCalibrator.fit(cal_confs, cal_labels, alpha_levels=(0.10,))
        cov = empirical_coverage(cal, eval_confs, eval_labels, alpha=0.10)
        # Nominal 0.90 per class. With cal n_pos ~390, n_neg ~110, bounds
        # are ~0.051 and ~0.095. Allow 2x bound for single-trial.
        assert (
            abs(cov["empirical_coverage_positive"] - 0.90)
            <= 2 * cov["finite_sample_bound_positive"]
        )
        assert (
            abs(cov["empirical_coverage_negative"] - 0.90)
            <= 2 * cov["finite_sample_bound_negative"]
        )

    def test_repeated_trials_coverage_holds(self):
        cov_positive = []
        cov_negative = []
        for seed in range(20):
            cal_confs, cal_labels = self._generate_exchangeable(n=300, seed=seed)
            eval_confs, eval_labels = self._generate_exchangeable(
                n=300, seed=seed + 100,
            )
            cal = ConformalCalibrator.fit(
                cal_confs, cal_labels, alpha_levels=(0.10,),
            )
            cov = empirical_coverage(cal, eval_confs, eval_labels, alpha=0.10)
            cov_positive.append(cov["empirical_coverage_positive"])
            cov_negative.append(cov["empirical_coverage_negative"])
        # Average across trials should be close to nominal 0.90.
        assert abs(np.mean(cov_positive) - 0.90) < 0.02
        assert abs(np.mean(cov_negative) - 0.90) < 0.05