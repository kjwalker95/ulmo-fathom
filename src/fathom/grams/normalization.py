"""Split-window normalization for spectrogram ambient subtraction.

Per-frequency-bin local-ambient estimation: for each bin, the ambient is the
mean over a `train` window EXCLUDING a `central` guard region around the bin.
Subtracting the ambient highlights tonal lines against background noise, matching
IUSS operator-display convention (Sprint1_Plan §3).

Two-Pass Split Window (TPSW) tuning lives in Sprint 2 (PCD v2 §6.3 Method A).
Sprint 1 uses the single-pass split-window form below.
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