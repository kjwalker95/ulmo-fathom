"""DeepShip ingestion module.

DeepShip is the Sprint 1 primary acoustic dataset. Two on-disk layouts are
supported:

1. Per-vessel directories: `<root>/<class>/<vessel>/<recording>.wav`.
   Vessel-level metadata is taken from the directory name.
2. Flat per-class: `<root>/<class>/<recording>.wav`.
   This is what the current DeepShip release ships. Per the DeepShip README,
   the dataset has 265 distinct ships across 4 classes, and per-class numeric
   filenames are unique vessel identifiers. The loader treats `recording_id`
   (filename stem) as `vessel_id` for this layout, preserving vessel-level
   metadata for Phase 1 splits.
"""
from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf

from ..audit import hash_file_sha256, now_utc
from ..models import DatasetIndex, RecordingMetadata

LOG = logging.getLogger(__name__)

DEEPSHIP_CLASSES = {"Cargo", "Passengership", "Tanker", "Tug"}
WAV_SUFFIXES = {".wav", ".flac"}


def _probe_recording(path: Path) -> tuple[int, float, int]:
    """Return (sample_rate_hz, duration_s, n_channels)."""
    info = sf.info(str(path))
    return info.samplerate, info.frames / info.samplerate, info.channels


def _audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in WAV_SUFFIXES)


def index_deepship(root: Path, min_duration_s: float = 3.0) -> DatasetIndex:
    """Walk the DeepShip release at `root` and produce a DatasetIndex."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"DeepShip root not found: {root}")

    recordings: list[RecordingMetadata] = []

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_label = class_dir.name
        if class_label not in DEEPSHIP_CLASSES:
            LOG.warning("unexpected class directory %s; including anyway", class_label)
        vessel_dirs = [p for p in class_dir.iterdir() if p.is_dir()]
        if vessel_dirs:
            for vessel_dir in sorted(vessel_dirs):
                vessel_id = vessel_dir.name
                for wav in _audio_files(vessel_dir):
                    rec = _probe_to_metadata(
                        wav,
                        min_duration_s=min_duration_s,
                        class_label=class_label,
                        vessel_id=vessel_id,
                    )
                    if rec is not None:
                        recordings.append(rec)
        else:
            LOG.info(
                "DeepShip class directory %s uses flat layout; treating recording_id "
                "as vessel_id (per DeepShip release convention)",
                class_dir.name,
            )
            for wav in _audio_files(class_dir):
                rec = _probe_to_metadata(
                    wav,
                    min_duration_s=min_duration_s,
                    class_label=class_label,
                    vessel_id=wav.stem,  # flat-layout convention: filename stem == vessel_id
                    note="vessel_id from filename (flat-layout DeepShip release)",
                )
                if rec is not None:
                    recordings.append(rec)

    LOG.info("indexed %d DeepShip recordings under %s", len(recordings), root)
    return DatasetIndex(
        dataset="deepship",
        root=root,
        recordings=recordings,
        index_built_at=now_utc(),
    )


def _probe_to_metadata(
    wav: Path,
    *,
    min_duration_s: float,
    class_label: str,
    vessel_id: str | None,
    note: str | None = None,
) -> RecordingMetadata | None:
    try:
        sr, dur, nch = _probe_recording(wav)
    except Exception:
        LOG.exception("failed to probe %s; skipping", wav)
        return None
    if dur < min_duration_s:
        return None
    return RecordingMetadata(
        recording_id=wav.stem,
        vessel_id=vessel_id,
        dataset="deepship",
        class_label=class_label,
        sample_rate_hz=sr,
        duration_s=dur,
        n_channels=nch,
        path=wav,
        notes=note,
    )


def write_manifest(index: DatasetIndex, out_path: Path) -> Path:
    """Write the index as JSON plus a SHA256 sidecar of the JSON content."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(index.model_dump_json(indent=2))
    digest = hash_file_sha256(out_path)
    sha_path = out_path.with_suffix(out_path.suffix + ".sha256")
    sha_path.write_text(digest + "\n")
    return out_path