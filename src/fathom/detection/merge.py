"""Post-hoc cluster-merge of nearby lines of interest.

Sprint 2 finding: STFT leakage from strong tonals splits one tonal into multiple
parallel lines along the leakage envelope (per the C3 smoke test, a synthetic
50 Hz tonal at amplitude 0.5 produced 4 lines at bins 22/24/27/28). On real
DeepShip recordings, the same effect appears at lower magnitude — multiple
closely-spaced lines reported for what an operator would read as a single tonal
contact.

This module coalesces nearby `LineOfInterest` results into a single representative
line, improving operator readability without changing the underlying detection
algorithm. UX fix, not a correctness fix. Off by default in legacy configs;
enabled in `configs/sprint3.yaml` via `DetectionConfig.merge_nearby_lines=True`.

Per Sprint3_Plan §3, this is a Tuor-product capability (operates on Tuor's
LineOfInterest output); future Fathom products with line-of-interest-shaped
output may reuse the same merge primitive or implement their own.
"""
from __future__ import annotations

import logging

from ..audit import new_correlation_id
from ..models import LineOfInterest

LOG = logging.getLogger(__name__)


def merge_nearby_lines(
    lines: list[LineOfInterest],
    *,
    merge_freq_tolerance_hz: float,
) -> list[LineOfInterest]:
    """Coalesce nearby lines whose time intervals overlap and whose frequencies
    fall within `merge_freq_tolerance_hz` of each other.

    The merged line takes:
    - frequency_hz from the max-SNR contributing line
    - bandwidth_hz covering the union of contributing freq extents
    - timestamp = min over contributors (start of union)
    - persistence_s = max(end) - min(start) across contributors
    - snr_db = max over contributors
    - detection_method, array_id, beam_id from the max-SNR contributor
    - fresh correlation_id (the merged line is a new entity)
    - confidence = None (calibrated uncertainty arrives in Phase 1)

    Iterative: re-merges until a pass produces no further changes (typically
    converges in <= 3 passes).
    """
    if merge_freq_tolerance_hz < 0:
        raise ValueError(
            f"merge_freq_tolerance_hz must be non-negative; got {merge_freq_tolerance_hz}"
        )
    if len(lines) < 2:
        return list(lines)

    current = sorted(lines, key=lambda l: (l.timestamp, l.frequency_hz))
    for _ in range(len(lines)):  # safety bound; converges in <= 3 passes in practice
        merged = _merge_pass(current, merge_freq_tolerance_hz)
        if len(merged) == len(current):
            return merged
        current = merged
    LOG.warning(
        "merge_nearby_lines did not converge after %d passes; returning current state",
        len(lines),
    )
    return current


def _merge_pass(
    lines_sorted: list[LineOfInterest],
    tol_hz: float,
) -> list[LineOfInterest]:
    """Single greedy left-to-right pass."""
    merged: list[LineOfInterest] = []
    for line in lines_sorted:
        if merged:
            last = merged[-1]
            if _can_merge(last, line, tol_hz):
                merged[-1] = _combine(last, line)
                continue
        merged.append(line)
    return merged


def _can_merge(a: LineOfInterest, b: LineOfInterest, tol_hz: float) -> bool:
    a_start = a.timestamp.timestamp()
    b_start = b.timestamp.timestamp()
    a_end = a_start + a.persistence_s
    b_end = b_start + b.persistence_s
    if a_end < b_start or b_end < a_start:
        return False
    return abs(a.frequency_hz - b.frequency_hz) <= tol_hz


def _combine(a: LineOfInterest, b: LineOfInterest) -> LineOfInterest:
    primary = a if a.snr_db >= b.snr_db else b

    start_ts = a.timestamp if a.timestamp <= b.timestamp else b.timestamp
    a_end_s = a.timestamp.timestamp() + a.persistence_s
    b_end_s = b.timestamp.timestamp() + b.persistence_s
    end_s = max(a_end_s, b_end_s)
    union_persistence_s = end_s - start_ts.timestamp()

    def edges(loi: LineOfInterest) -> tuple[float, float]:
        bw = loi.bandwidth_hz if loi.bandwidth_hz is not None else 0.0
        return (loi.frequency_hz - bw / 2, loi.frequency_hz + bw / 2)

    a_lo, a_hi = edges(a)
    b_lo, b_hi = edges(b)
    union_bw = max(a_hi, b_hi) - min(a_lo, b_lo)

    return LineOfInterest(
        correlation_id=new_correlation_id(),
        array_id=primary.array_id,
        beam_id=primary.beam_id,
        timestamp=start_ts,
        frequency_hz=primary.frequency_hz,
        bandwidth_hz=union_bw,
        snr_db=max(a.snr_db, b.snr_db),
        persistence_s=union_persistence_s,
        detection_method=primary.detection_method,
        confidence=None,
    )