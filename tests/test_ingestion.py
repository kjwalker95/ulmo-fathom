"""Smoke tests for ingestion modules using synthetic WAV files."""
from pathlib import Path

import numpy as np
import soundfile as sf

from fathom.ingestion.deepship import index_deepship, write_manifest


def _write_synthetic(path: Path, duration_s: float = 4.0, sr: int = 32000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sr * duration_s)) / sr
    sig = 0.1 * np.sin(2 * np.pi * 50 * t).astype("float32")
    sf.write(str(path), sig, sr)


def test_index_deepship_per_vessel_layout(tmp_path: Path):
    root = tmp_path / "deepship"
    _write_synthetic(root / "Cargo" / "vesselA" / "rec0001.wav")
    _write_synthetic(root / "Cargo" / "vesselA" / "rec0002.wav")
    _write_synthetic(root / "Tug" / "vesselB" / "rec0003.wav")
    index = index_deepship(root)
    assert index.dataset == "deepship"
    assert len(index.recordings) == 3
    classes = {r.class_label for r in index.recordings}
    assert classes == {"Cargo", "Tug"}
    assert {r.vessel_id for r in index.recordings} == {"vesselA", "vesselB"}


def test_index_deepship_flat_layout_uses_recording_id_as_vessel_id(tmp_path: Path, caplog):
    """The DeepShip release ships flat. Each numeric .wav is a distinct vessel
    (DeepShip has 265 ships across 4 classes; per-class numeric IDs are unique)."""
    root = tmp_path / "deepship"
    _write_synthetic(root / "Cargo" / "103.wav")
    _write_synthetic(root / "Cargo" / "110.wav")
    with caplog.at_level("INFO"):
        index = index_deepship(root)
    assert any("flat layout" in m for m in caplog.messages)
    vessel_ids = {r.vessel_id for r in index.recordings}
    assert vessel_ids == {"103", "110"}
    # vessel_id == recording_id under this convention
    assert all(r.vessel_id == r.recording_id for r in index.recordings)


def test_write_manifest_produces_sidecar(tmp_path: Path):
    root = tmp_path / "deepship"
    _write_synthetic(root / "Cargo" / "vesselA" / "rec0001.wav")
    index = index_deepship(root)
    manifest_path = write_manifest(index, tmp_path / "manifest.json")
    assert manifest_path.exists()
    sha_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    assert sha_path.exists()
    assert len(sha_path.read_text().strip()) == 64  # SHA256 hex