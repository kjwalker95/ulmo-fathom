"""ShipsEar ingestion module.

Best-guess parsing of the ShipsEar release filename convention. Reconciles when
the data lands and we can verify the actual layout (Sprint1_Plan §8 risk #2).
ShipsEar is optional for Sprint 1.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import soundfile as sf

from ..audit import now_utc
from ..models import DatasetIndex, RecordingMetadata

LOG = logging.getLogger(__name__)

WAV_SUFFIXES = {".wav", ".flac"}

# ShipsEar published filename pattern (best-guess; reconcile when data lands):
#   <id>__<class>__<vessel-or-context>__<datetime>.wav
SHIPSEAR_FILENAME_RE = re.compile(
    r"^(?P<rid>\d+)[_-]+(?P<class>[A-Za-z]+)[_-]+(?P<vessel>[A-Za-z0-9]+)",
)


def _parse_shipsear_filename(name: str) -> tuple[str | None, str | None]:
    m = SHIPSEAR_FILENAME_RE.match(name)
    if not m:
        return None, None
    return m.group("class"), m.group("vessel")


def index_shipsear(root: Path, min_duration_s: float = 3.0) -> DatasetIndex:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"ShipsEar root not found: {root}")
    recordings: list[RecordingMetadata] = []
    for wav in sorted(root.rglob("*")):
        if not wav.is_file() or wav.suffix.lower() not in WAV_SUFFIXES:
            continue
        try:
            info = sf.info(str(wav))
        except Exception:
            LOG.exception("failed to probe %s; skipping", wav)
            continue
        dur = info.frames / info.samplerate
        if dur < min_duration_s:
            continue
        class_label, vessel_id = _parse_shipsear_filename(wav.stem)
        if class_label is None:
            LOG.warning("could not parse ShipsEar filename %s; class/vessel unavailable", wav.name)
        recordings.append(
            RecordingMetadata(
                recording_id=wav.stem,
                vessel_id=vessel_id,
                dataset="shipsear",
                class_label=class_label,
                sample_rate_hz=info.samplerate,
                duration_s=dur,
                n_channels=info.channels,
                path=wav,
                notes=None if class_label else "filename pattern not matched",
            )
        )
    LOG.info("indexed %d ShipsEar recordings under %s", len(recordings), root)
    return DatasetIndex(dataset="shipsear", root=root, recordings=recordings, index_built_at=now_utc())