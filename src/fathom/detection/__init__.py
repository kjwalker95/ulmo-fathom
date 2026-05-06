"""Classical line detection (Sprint 2).

Public API:
- `detect_lines`, `DetectionConfig` — top-level orchestrator (lines.py).
- `detect_peaks_per_bin`, `detect_peaks_2d` — peak detectors over TPSW-normalized
  power-dB grams (peaks.py).
- `PersistenceConfig`, `PersistentLine`, `filter_persistent_lines` — persistence
  filter that aggregates per-cell peaks into operationally-shaped lines
  (persistence.py).
"""
from .lines import DetectionConfig, detect_lines
from .peaks import detect_peaks_2d, detect_peaks_per_bin
from .persistence import PersistenceConfig, PersistentLine, filter_persistent_lines

__all__ = [
    "detect_lines",
    "DetectionConfig",
    "detect_peaks_per_bin",
    "detect_peaks_2d",
    "PersistenceConfig",
    "PersistentLine",
    "filter_persistent_lines",
]