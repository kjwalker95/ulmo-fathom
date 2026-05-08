"""Cluster-merge tests (Sprint 3 Cluster 2)."""
from datetime import datetime, timedelta, timezone

import pytest

from fathom.detection import merge_nearby_lines
from fathom.models import DetectionMethod, LineOfInterest

T0 = datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)


def _line(freq_hz, t_offset_s, persistence_s, snr_db=10.0, bw_hz=2.0):
    return LineOfInterest(
        correlation_id=f"test-{int(freq_hz * 1000)}-{int(t_offset_s * 1000)}",
        array_id="TEST",
        beam_id=None,
        timestamp=T0 + timedelta(seconds=t_offset_s),
        frequency_hz=freq_hz,
        bandwidth_hz=bw_hz,
        snr_db=snr_db,
        persistence_s=persistence_s,
        detection_method=DetectionMethod.CLASSICAL,
        confidence=None,
    )


def test_merge_two_overlapping_close_lines():
    a = _line(freq_hz=49.0, t_offset_s=0.0, persistence_s=4.0, snr_db=15.0)
    b = _line(freq_hz=51.0, t_offset_s=1.0, persistence_s=4.0, snr_db=12.0)
    out = merge_nearby_lines([a, b], merge_freq_tolerance_hz=3.0)
    assert len(out) == 1
    merged = out[0]
    assert merged.frequency_hz == 49.0  # max-SNR member's freq (a.snr=15 > b.snr=12)
    assert merged.snr_db == 15.0
    assert abs(merged.persistence_s - 5.0) < 1e-9  # union: 0 -> max(4, 1+4) = 5
    # bandwidth covers [48, 52] with bw=2 each side -> 4 Hz
    assert abs(merged.bandwidth_hz - 4.0) < 0.01


def test_no_merge_far_apart_in_freq():
    a = _line(freq_hz=50.0, t_offset_s=0.0, persistence_s=4.0)
    b = _line(freq_hz=200.0, t_offset_s=0.0, persistence_s=4.0)
    out = merge_nearby_lines([a, b], merge_freq_tolerance_hz=5.0)
    assert len(out) == 2


def test_no_merge_non_overlapping_time():
    a = _line(freq_hz=50.0, t_offset_s=0.0, persistence_s=2.0)   # ends at t=2
    b = _line(freq_hz=51.0, t_offset_s=10.0, persistence_s=2.0)  # starts at t=10
    out = merge_nearby_lines([a, b], merge_freq_tolerance_hz=5.0)
    assert len(out) == 2


def test_iterative_chain_merge():
    """A close to B, B close to C, A far from C -> A-B-C all merge transitively."""
    a = _line(freq_hz=48.0, t_offset_s=0.0, persistence_s=4.0, snr_db=10.0)
    b = _line(freq_hz=50.0, t_offset_s=0.5, persistence_s=4.0, snr_db=12.0)
    c = _line(freq_hz=52.0, t_offset_s=1.0, persistence_s=4.0, snr_db=11.0)
    out = merge_nearby_lines([a, b, c], merge_freq_tolerance_hz=3.0)
    assert len(out) == 1


def test_empty_input():
    assert merge_nearby_lines([], merge_freq_tolerance_hz=5.0) == []


def test_single_line_unchanged():
    a = _line(freq_hz=50.0, t_offset_s=0.0, persistence_s=4.0)
    out = merge_nearby_lines([a], merge_freq_tolerance_hz=5.0)
    assert out == [a]


def test_negative_tolerance_rejected():
    a = _line(freq_hz=50.0, t_offset_s=0.0, persistence_s=4.0)
    b = _line(freq_hz=51.0, t_offset_s=0.0, persistence_s=4.0)
    with pytest.raises(ValueError, match="non-negative"):
        merge_nearby_lines([a, b], merge_freq_tolerance_hz=-1.0)