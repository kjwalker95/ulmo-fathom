"""Top-level classical line-detection orchestrator.

Ties TPSW two-pass normalization (grams.normalization), peak detection
(detection.peaks), persistence filtering (detection.persistence), and optional
post-hoc cluster-merge (detection.merge) into a single `detect_lines(gram, config, ...)`
call. Populates LineOfInterest records, publishes Topic.LINE_DETECTED on the
event bus, returns the list.

Per Sprint2_Plan §3 / Sprint3_Plan §3, the library function's side-effect
surface is bounded to event publishing; lines.jsonl writing and audit sidecars
live in the sanity-check script, not here. Sprint 3 added the optional
cluster-merge pass; events are published post-merge so consumers see the
operator-shaped output.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from ..audit import new_correlation_id
from ..events import EventBus, Topic, get_default_bus
from ..grams.lofar import LOFARGram
from ..grams.normalization import tpsw_normalize
from ..models import DetectionMethod, LineOfInterest
from .merge import merge_nearby_lines
from .peaks import detect_peaks_2d, detect_peaks_per_bin
from .persistence import PersistenceConfig, filter_persistent_lines

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionConfig:
    """Sprint 2 classical line-detection parameters + Sprint 3 cluster-merge.

    Carried verbatim into audit sidecars as part of the parameter snapshot so
    detection output is reproducible from (recording, gram config, this config).
    """
    # TPSW
    tpsw_first_pass_threshold_db: float = 6.0
    tpsw_min_unmasked_train_bins: int = 16
    # peak detection
    peak_method: Literal["per_bin", "two_d"] = "per_bin"
    peak_snr_threshold_db: float = 8.0
    peak_min_separation_time_bins: int = 4
    peak_two_d_neighborhood: tuple[int, int] = field(default=(3, 5))
    # persistence
    min_persistence_s: float = 3.0
    frequency_drift_bins: int = 2
    gap_tolerance_time_bins: int = 8
    # cluster-merge (Sprint 3 Cluster 2). Off by default for backward-compat with
    # configs/sprint2.yaml; configs/sprint3.yaml turns this on.
    merge_nearby_lines: bool = False
    # Default tolerance derived from drift_bins x bin_width when None.
    merge_freq_tolerance_hz: float | None = None


def detect_lines(
    gram: LOFARGram,
    config: DetectionConfig,
    *,
    array_id: str,
    beam_id: str | None,
    recording_start_utc: datetime,
    bus: EventBus | None = None,
) -> list[LineOfInterest]:
    """Run the classical line-detection pipeline over a LOFAR gram.

    1. Re-derive a TPSW-normalized power-dB array from `gram.power_db`.
    2. Run the configured peak detector (`per_bin` or `two_d`).
    3. Aggregate peaks into PersistentLines via the persistence filter.
    4. Build a LineOfInterest for each PersistentLine.
    5. If `config.merge_nearby_lines`, coalesce nearby lines (Sprint 3 UX fix).
    6. Publish each (post-merge) LineOfInterest on Topic.LINE_DETECTED.
    """
    # 1. TPSW two-pass.
    tpsw_db = tpsw_normalize(
        gram.power_db,
        train_window_bins=gram.config.normalization_train_window_bins,
        central_window_bins=gram.config.normalization_central_window_bins,
        gap_bins=gram.config.normalization_gap_bins,
        first_pass_threshold_db=config.tpsw_first_pass_threshold_db,
        min_unmasked_train_bins=config.tpsw_min_unmasked_train_bins,
        axis=0,
    )

    # 2. Peaks.
    if config.peak_method == "per_bin":
        peak_mask = detect_peaks_per_bin(
            tpsw_db,
            snr_threshold_db=config.peak_snr_threshold_db,
            min_separation_time_bins=config.peak_min_separation_time_bins,
        )
    elif config.peak_method == "two_d":
        peak_mask = detect_peaks_2d(
            tpsw_db,
            snr_threshold_db=config.peak_snr_threshold_db,
            neighborhood_size=config.peak_two_d_neighborhood,
        )
    else:
        raise ValueError(f"unknown peak_method: {config.peak_method!r}")

    # 3. Persistence filter.
    if len(gram.times_s) >= 2:
        hop_length_s = float(gram.times_s[1] - gram.times_s[0])
    else:
        hop_length_s = float(gram.config.stft.hop_length) / float(gram.config.stft.sample_rate)
    persistence_cfg = PersistenceConfig(
        min_persistence_s=config.min_persistence_s,
        frequency_drift_bins=config.frequency_drift_bins,
        gap_tolerance_time_bins=config.gap_tolerance_time_bins,
        hop_length_s=hop_length_s,
    )
    persistent = filter_persistent_lines(peak_mask, tpsw_db, persistence_cfg)

    # 4. LineOfInterest assembly.
    lines: list[LineOfInterest] = []
    for pl in persistent:
        t_start_s = float(gram.times_s[pl.t_start_bin])
        t_end_s = float(gram.times_s[pl.t_end_bin])
        f_min = float(gram.frequencies_hz[pl.freq_bin_min])
        f_max = float(gram.frequencies_hz[pl.freq_bin_max])
        loi = LineOfInterest(
            correlation_id=new_correlation_id(),
            array_id=array_id,
            beam_id=beam_id,
            timestamp=recording_start_utc + timedelta(seconds=t_start_s),
            frequency_hz=float(gram.frequencies_hz[pl.freq_bin_center]),
            bandwidth_hz=f_max - f_min,
            snr_db=float(pl.peak_snr_db),
            persistence_s=t_end_s - t_start_s,
            detection_method=DetectionMethod.CLASSICAL,
            confidence=None,
        )
        lines.append(loi)

    # 5. Optional cluster-merge (Sprint 3).
    if config.merge_nearby_lines:
        if config.merge_freq_tolerance_hz is not None:
            tol_hz = config.merge_freq_tolerance_hz
        elif len(gram.frequencies_hz) >= 2:
            bin_width_hz = float(gram.frequencies_hz[1] - gram.frequencies_hz[0])
            tol_hz = 2.0 * config.frequency_drift_bins * bin_width_hz
        else:
            tol_hz = 0.0
        pre = len(lines)
        lines = merge_nearby_lines(lines, merge_freq_tolerance_hz=tol_hz)
        if pre != len(lines):
            LOG.info("cluster-merge: %d -> %d lines (tol=%.2f Hz)", pre, len(lines), tol_hz)

    # 6. Publish (post-merge).
    publish_bus = bus if bus is not None else get_default_bus()
    for loi in lines:
        publish_bus.publish(Topic.LINE_DETECTED, loi)

    LOG.info("detected %d line(s) on array %s", len(lines), array_id)
    return lines