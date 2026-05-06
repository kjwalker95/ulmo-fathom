"""Persistence filter: aggregates per-cell peaks into operationally-shaped lines.

A "line of interest" is a sequence of peaks at roughly the same frequency that
lasts at least `min_persistence_s` seconds. Frequency drift across consecutive
peaks is bounded by `frequency_drift_bins`; brief intermittencies up to
`gap_tolerance_time_bins` are bridged. Sub-persistence noise transients are
rejected (Sprint2_Plan §3).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersistenceConfig:
    min_persistence_s: float
    frequency_drift_bins: int
    gap_tolerance_time_bins: int
    hop_length_s: float

    def __post_init__(self) -> None:
        if self.hop_length_s <= 0:
            raise ValueError(f"hop_length_s must be positive; got {self.hop_length_s}")
        if self.min_persistence_s < 0:
            raise ValueError(f"min_persistence_s must be non-negative; got {self.min_persistence_s}")
        if self.frequency_drift_bins < 0:
            raise ValueError(
                f"frequency_drift_bins must be non-negative; got {self.frequency_drift_bins}"
            )
        if self.gap_tolerance_time_bins < 0:
            raise ValueError(
                f"gap_tolerance_time_bins must be non-negative; got {self.gap_tolerance_time_bins}"
            )


@dataclass(frozen=True)
class PersistentLine:
    freq_bin_center: int   # SNR-weighted mean frequency bin across the run
    freq_bin_min: int
    freq_bin_max: int
    t_start_bin: int       # first peak's time bin
    t_end_bin: int         # last peak's time bin
    peak_snr_db: float     # max SNR (dB above local ambient) across the run
    n_peaks: int


def filter_persistent_lines(
    peak_mask: np.ndarray,
    tpsw_db: np.ndarray,
    config: PersistenceConfig,
) -> list[PersistentLine]:
    """Greedy left-to-right aggregation of peaks into PersistentLines.

    Algorithm (per Sprint2_Plan §3):
    - Peaks are visited in (time, frequency) order.
    - Each unconsumed peak starts a candidate run.
    - The run grows forward in time: at each successive time bin (up to
      `gap_tolerance_time_bins` empty bins), the closest unconsumed peak
      within `frequency_drift_bins` of the run's current frequency is added.
      The run's "current frequency" updates step-by-step so slow drift is
      captured naturally.
    - A run is reported as a PersistentLine iff its time span is at least
      `min_persistence_s` seconds.
    """
    if peak_mask.shape != tpsw_db.shape:
        raise ValueError(
            f"shape mismatch: peak_mask {peak_mask.shape} vs tpsw_db {tpsw_db.shape}"
        )
    n_freq, n_time = peak_mask.shape

    peak_coords = np.argwhere(peak_mask)  # rows of (freq_bin, time_bin)
    if len(peak_coords) == 0:
        return []

    # Sort by time, then frequency (lexsort: last key is primary).
    order = np.lexsort((peak_coords[:, 0], peak_coords[:, 1]))
    peak_coords = peak_coords[order]

    # Index peaks by time bin for fast forward lookup during run growth.
    time_index: dict[int, list[int]] = {}
    for k, coord in enumerate(peak_coords):
        time_index.setdefault(int(coord[1]), []).append(k)

    consumed = np.zeros(len(peak_coords), dtype=bool)
    min_persistence_bins = config.min_persistence_s / config.hop_length_s

    lines: list[PersistentLine] = []

    for k_start in range(len(peak_coords)):
        if consumed[k_start]:
            continue
        i_start, j_start = (int(peak_coords[k_start, 0]), int(peak_coords[k_start, 1]))
        run_indices = [k_start]
        consumed[k_start] = True
        current_freq = i_start
        last_time = j_start

        t = last_time + 1
        while t < n_time and (t - last_time) <= config.gap_tolerance_time_bins:
            best_k = -1
            best_dist = config.frequency_drift_bins + 1
            for k_cand in time_index.get(t, []):
                if consumed[k_cand]:
                    continue
                dist = abs(int(peak_coords[k_cand, 0]) - current_freq)
                if dist <= config.frequency_drift_bins and dist < best_dist:
                    best_k = k_cand
                    best_dist = dist
            if best_k >= 0:
                run_indices.append(best_k)
                consumed[best_k] = True
                current_freq = int(peak_coords[best_k, 0])
                last_time = t
            t += 1

        # Compute run statistics.
        run_freqs = peak_coords[run_indices, 0].astype(int)
        run_times = peak_coords[run_indices, 1].astype(int)
        run_snrs = tpsw_db[run_freqs, run_times]

        t_start = int(run_times.min())
        t_end = int(run_times.max())
        duration_bins = t_end - t_start

        if duration_bins < min_persistence_bins:
            continue  # too short — peaks stay consumed; later starts won't reabsorb

        weights = np.maximum(run_snrs, 0.0)
        if float(weights.sum()) > 0.0:
            freq_center = int(round(float(np.average(run_freqs, weights=weights))))
        else:
            freq_center = int(round(float(run_freqs.mean())))

        lines.append(
            PersistentLine(
                freq_bin_center=freq_center,
                freq_bin_min=int(run_freqs.min()),
                freq_bin_max=int(run_freqs.max()),
                t_start_bin=t_start,
                t_end_bin=t_end,
                peak_snr_db=float(run_snrs.max()),
                n_peaks=len(run_indices),
            )
        )

    return lines