"""Per-bin and 2D peak detection over TPSW-normalized LOFAR grams.

Both detectors consume a TPSW-normalized power-dB array of shape (n_freq, n_time)
and return a boolean detection mask of the same shape. Peak detectors are
intentionally cheap; the persistence filter (persistence.py) is what turns a
"cell exceeded threshold once" event into "this is a sustained tonal worth
reporting" (Sprint2_Plan §3).
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks

LOG = logging.getLogger(__name__)


def detect_peaks_per_bin(
    tpsw_db: np.ndarray,
    *,
    snr_threshold_db: float = 8.0,
    min_separation_time_bins: int = 4,
) -> np.ndarray:
    """Per-frequency-bin time-direction local maxima above an SNR threshold.

    For each frequency row, scipy.signal.find_peaks finds local maxima above
    `snr_threshold_db` with `min_separation_time_bins` minimum spacing. Returns
    a boolean mask of the same shape as `tpsw_db`.
    """
    if tpsw_db.ndim != 2:
        raise ValueError(f"expected 2D (n_freq, n_time); got shape {tpsw_db.shape}")
    distance = max(1, int(min_separation_time_bins))
    mask = np.zeros_like(tpsw_db, dtype=bool)
    for i in range(tpsw_db.shape[0]):
        peaks, _ = find_peaks(tpsw_db[i], height=snr_threshold_db, distance=distance)
        if len(peaks) > 0:
            mask[i, peaks] = True
    return mask


def detect_peaks_2d(
    tpsw_db: np.ndarray,
    *,
    snr_threshold_db: float = 8.0,
    neighborhood_size: tuple[int, int] = (3, 5),
) -> np.ndarray:
    """2D local-maximum detector via scipy.ndimage.maximum_filter.

    A cell is a peak iff it equals the maximum in its (freq, time) neighborhood
    AND exceeds `snr_threshold_db`. Flat plateaus mark every cell as a peak;
    that is acceptable for Sprint 2 (the persistence filter aggregates anyway).
    """
    if tpsw_db.ndim != 2:
        raise ValueError(f"expected 2D (n_freq, n_time); got shape {tpsw_db.shape}")
    if len(neighborhood_size) != 2:
        raise ValueError(f"neighborhood_size must be (freq, time); got {neighborhood_size}")
    local_max = maximum_filter(tpsw_db, size=neighborhood_size, mode="nearest")
    return (tpsw_db == local_max) & (tpsw_db >= snr_threshold_db)