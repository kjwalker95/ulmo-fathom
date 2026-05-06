"""Split-window and TPSW normalization for spectrogram ambient subtraction.

Single-pass split-window: per-frequency-bin local-ambient estimation. For each
bin, the ambient is the mean over a `train` window EXCLUDING a `central` guard
region around the bin. Subtracting the ambient highlights tonal lines against
background noise, matching IUSS operator-display convention (Sprint1_Plan §3).

Two-Pass Split Window (TPSW): a strong tonal contaminates the single-pass
ambient estimate at neighboring frequency bins (the tonal sits inside their
training windows). TPSW addresses this with a second pass that masks
candidate-signal cells out of the training window, producing a cleaner local
ambient and a higher effective SNR for detection (Sprint2_Plan §3, PCD v2 §6.3
Method A). Sprint 2 detection consumes the TPSW output; rendering still uses
the single-pass form for display continuity with Sprint 1.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def split_window_normalize(
    power: np.ndarray,
    train_window_bins: int = 33,
    central_window_bins: int = 5,
    gap_bins: int = 1,
    axis: int = 0,
) -> np.ndarray:
    """Subtract a local ambient estimate computed over a training window
    that excludes a central guard region around each bin.

    `axis` is the axis along which to estimate ambient; default 0 (frequency).
    Implementation: ambient[i] = (sum_train[i] - sum_central[i]) / (n_train - n_central),
    where sum_X[i] is uniform_filter1d(power, size=X) * X.
    """
    central_total_bins = central_window_bins + 2 * gap_bins
    if train_window_bins <= central_total_bins:
        raise ValueError(
            f"train_window_bins ({train_window_bins}) must exceed "
            f"central_window_bins + 2*gap_bins ({central_total_bins})"
        )
    train_mean = uniform_filter1d(power, size=train_window_bins, axis=axis, mode="nearest")
    central_mean = uniform_filter1d(power, size=central_total_bins, axis=axis, mode="nearest")
    n_train_only = train_window_bins - central_total_bins
    ambient = (train_mean * train_window_bins - central_mean * central_total_bins) / n_train_only
    return power - ambient


def tpsw_normalize(
    power_db: np.ndarray,
    *,
    train_window_bins: int = 33,
    central_window_bins: int = 5,
    gap_bins: int = 1,
    first_pass_threshold_db: float = 6.0,
    min_unmasked_train_bins: int = 16,
    axis: int = 0,
) -> np.ndarray:
    """Two-Pass Split Window normalization.

    Pass 1: standard split-window (computed inline so we have access to the
    first-pass ambient as a fallback, not just the residual).
    Pass 2: re-estimate ambient with cells exceeding `first_pass_threshold_db`
    in the first-pass residual masked out of the training window. Where the
    unmasked training-ring count drops below `min_unmasked_train_bins`, fall
    back to the first-pass ambient at that cell.

    Returns power_db minus the second-pass ambient estimate (with first-pass
    fallback applied where overmasked).
    """
    central_total_bins = central_window_bins + 2 * gap_bins
    if train_window_bins <= central_total_bins:
        raise ValueError(
            f"train_window_bins ({train_window_bins}) must exceed "
            f"central_window_bins + 2*gap_bins ({central_total_bins})"
        )
    n_train_only = train_window_bins - central_total_bins

    # Pass 1: derive first-pass ambient and residual.
    train_mean = uniform_filter1d(power_db, size=train_window_bins, axis=axis, mode="nearest")
    central_mean = uniform_filter1d(power_db, size=central_total_bins, axis=axis, mode="nearest")
    first_pass_ambient = (
        train_mean * train_window_bins - central_mean * central_total_bins
    ) / n_train_only
    first_pass_residual = power_db - first_pass_ambient

    # Candidate-signal mask: cells more than first_pass_threshold_db above local ambient.
    mask = first_pass_residual > first_pass_threshold_db

    # Pass 2: ring sums and counts with masked cells excluded.
    valid = (~mask).astype(np.float64)
    power_masked = np.where(mask, 0.0, power_db.astype(np.float64))

    train_sum = (
        uniform_filter1d(power_masked, size=train_window_bins, axis=axis, mode="nearest")
        * train_window_bins
    )
    central_sum = (
        uniform_filter1d(power_masked, size=central_total_bins, axis=axis, mode="nearest")
        * central_total_bins
    )
    train_count = (
        uniform_filter1d(valid, size=train_window_bins, axis=axis, mode="nearest")
        * train_window_bins
    )
    central_count = (
        uniform_filter1d(valid, size=central_total_bins, axis=axis, mode="nearest")
        * central_total_bins
    )

    ring_sum = train_sum - central_sum
    ring_count = train_count - central_count

    safe_ring_count = np.where(ring_count > 0, ring_count, 1.0)
    second_pass_ambient = ring_sum / safe_ring_count

    sufficient = ring_count >= min_unmasked_train_bins
    ambient = np.where(sufficient, second_pass_ambient, first_pass_ambient)

    return power_db - ambient