"""Unit tests for fathom.calibration.ensemble (Sprint 6 Cluster C)."""
import numpy as np
import pytest

from fathom.calibration.ensemble import (
    max_disagreement_patch_score,
    mean_prediction,
    member_disagreement_variance,
    mutual_information,
    patch_confidence,
    predictive_entropy,
)


def _all_high(n=5, h=4, w=4):
    return np.full((n, h, w), 0.95, dtype=np.float32)


def _all_low(n=5, h=4, w=4):
    return np.full((n, h, w), 0.05, dtype=np.float32)


def _split_3_high_2_low(h=4, w=4):
    """3 members at 0.98, 2 at 0.02. Disagreement at every pixel."""
    masks = np.zeros((5, h, w), dtype=np.float32)
    masks[:3] = 0.98
    masks[3:] = 0.02
    return masks


def _localized_disagreement(h=4, w=4):
    """All members agree (low) everywhere EXCEPT center pixel which splits."""
    masks = np.full((5, h, w), 0.05, dtype=np.float32)
    masks[:3, h // 2, w // 2] = 0.98
    masks[3:, h // 2, w // 2] = 0.02
    return masks


class TestMeanPrediction:
    def test_all_high(self):
        out = mean_prediction(_all_high())
        assert out.shape == (4, 4)
        assert np.allclose(out, 0.95)

    def test_split_3_2(self):
        out = mean_prediction(_split_3_high_2_low())
        # (3*0.98 + 2*0.02) / 5 = 0.596
        assert np.allclose(out, 0.596, atol=1e-4)

    def test_shape_check(self):
        with pytest.raises(ValueError, match="3D"):
            mean_prediction(np.zeros((4, 4)))


class TestPredictiveEntropy:
    def test_high_confidence_low_entropy(self):
        out = predictive_entropy(_all_high())
        # H(0.95) ~= 0.198 nats
        assert (out < 0.25).all()

    def test_split_high_entropy(self):
        out = predictive_entropy(_split_3_high_2_low())
        # H(0.596) ~= 0.674 nats
        assert (out > 0.6).all()
        assert (out < 0.7).all()


class TestMutualInformation:
    def test_all_agree_zero_mi(self):
        out = mutual_information(_all_high())
        assert (np.abs(out) < 1e-5).all()

    def test_split_positive_mi(self):
        # Members individually confident (low per-member entropy) but
        # ensemble mean near 0.5 (high mean entropy) -> positive MI.
        out = mutual_information(_split_3_high_2_low())
        assert (out > 0.4).all()

    def test_mi_non_negative_random(self):
        rng = np.random.default_rng(20260523)
        masks = rng.random((5, 8, 8), dtype=np.float64).astype(np.float32)
        out = mutual_information(masks)
        assert (out >= -1e-6).all()


class TestMemberDisagreementVariance:
    def test_all_agree_zero_var(self):
        out = member_disagreement_variance(_all_high())
        assert (np.abs(out) < 1e-6).all()

    def test_split_high_var(self):
        out = member_disagreement_variance(_split_3_high_2_low())
        # Pop var = (3*(0.98-0.596)^2 + 2*(0.02-0.596)^2) / 5 ~= 0.221
        assert np.allclose(out, 0.2212, atol=1e-3)


class TestMaxDisagreementPatchScore:
    def test_returns_scalar(self):
        score = max_disagreement_patch_score(_split_3_high_2_low())
        assert isinstance(score, float)

    def test_localized_max(self):
        # Disagreement only at center pixel -> max equals that pixel's var.
        score = max_disagreement_patch_score(_localized_disagreement())
        assert np.isclose(score, 0.2212, atol=1e-3)

    def test_all_agree_zero_score(self):
        score = max_disagreement_patch_score(_all_high())
        assert np.isclose(score, 0.0, atol=1e-6)


class TestPatchConfidence:
    def test_max_mean_all_high(self):
        score = patch_confidence(_all_high(), method="max_mean")
        assert np.isclose(score, 0.95, atol=1e-4)

    def test_max_mean_re_saturates_on_single_pixel(self):
        # The point: even though most pixels are low, ONE ensemble-mean
        # pixel near max -> re-saturates. This is why max_mean preserves
        # bimodal saturation.
        masks = np.full((5, 4, 4), 0.02, dtype=np.float32)
        masks[:, 0, 0] = 0.95
        score = patch_confidence(masks, method="max_mean")
        assert np.isclose(score, 0.95, atol=1e-4)

    def test_mean_max_robust_to_single_member_outlier(self):
        masks = np.full((5, 4, 4), 0.02, dtype=np.float32)
        masks[0, 0, 0] = 0.95  # only member 0 sees a hot pixel
        # member 0 max = 0.95; members 1-4 max = 0.02
        # mean_max = (0.95 + 4*0.02) / 5 = 0.206
        score = patch_confidence(masks, method="mean_max")
        assert np.isclose(score, 0.206, atol=1e-3)

    def test_peak_freq_band_uses_freq_axis(self):
        # Row 0 (freq=0) is hot for all time bins.
        masks = np.full((1, 4, 4), 0.02, dtype=np.float32)
        masks[0, 0, :] = 0.9
        score = patch_confidence(masks, method="peak_freq_band")
        # For each time bin, max-over-freq = 0.9. Mean over time = 0.9.
        assert np.isclose(score, 0.9, atol=1e-4)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown patch_confidence method"):
            patch_confidence(_all_high(), method="bogus")

    

from fathom.calibration.ensemble import compute_reliability_bins


class TestComputeReliabilityBins:
    def test_perfect_calibration_zero_ece(self):
        # 10 patches at p=0.05, 1 positive (10% rate matches bin 0 center).
        # 10 patches at p=0.95, 9 positives (90% rate matches bin 9 center).
        pred = np.array([0.05] * 10 + [0.95] * 10)
        truth = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        out = compute_reliability_bins(pred, truth, n_bins=10)
        assert out["n_patches"] == 20
        assert out["n_positives"] == 10
        # Gap in bin 0: |0.05 - 0.10| = 0.05; in bin 9: |0.95 - 0.90| = 0.05.
        # ECE = 0.5 * 0.05 + 0.5 * 0.05 = 0.05.
        assert np.isclose(out["ece"], 0.05, atol=1e-6)

    def test_bimodal_saturation_pattern(self):
        # Mimics Sprint 5 C5 finding: mass only at bin 0 and bin 9.
        pred = np.array([0.003] * 64 + [0.999] * 236)
        truth = np.array([1] * 11 + [0] * 53 + [1] * 223 + [0] * 13)
        out = compute_reliability_bins(pred, truth, n_bins=10)
        # Bins 1-8 all empty.
        for i in range(1, 9):
            assert out["bins"][i]["n_samples"] == 0
        # Bin 0 has 64 samples; bin 9 has 236.
        assert out["bins"][0]["n_samples"] == 64
        assert out["bins"][9]["n_samples"] == 236

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_reliability_bins(np.array([0.5]), np.array([1, 0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_reliability_bins(np.array([]), np.array([]))

    def test_n_positives_count(self):
        pred = np.array([0.5, 0.5, 0.5, 0.5])
        truth = np.array([1, 1, 1, 0])
        out = compute_reliability_bins(pred, truth, n_bins=10)
        assert out["n_positives"] == 3