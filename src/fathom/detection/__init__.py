"""Classical line detection (Sprint 2).

Public API:
- `detect_peaks_per_bin`, `detect_peaks_2d` — peak detectors over TPSW-normalized
  power-dB grams (peaks.py).
- `PersistenceConfig`, `PersistentLine`, `filter_persistent_lines` — persistence
  filter that aggregates per-cell peaks into operationally-shaped lines
  (persistence.py).
- `DetectionConfig`, `detect_lines` — Sprint 2 Cluster 3; orchestrator that ties
  TPSW + peaks + persistence together and emits LineOfInterest / publishes
  Topic.LINE_DETECTED (lines.py).
"""
from .peaks import detect_peaks_2d, detect_peaks_per_bin
from .persistence import PersistenceConfig, PersistentLine, filter_persistent_lines

__all__ = [
    "detect_peaks_per_bin",
    "detect_peaks_2d",
    "PersistenceConfig",
    "PersistentLine",
    "filter_persistent_lines",
]