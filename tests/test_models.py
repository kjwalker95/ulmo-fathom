"""Smoke tests for Pydantic data models."""
from datetime import datetime, timezone
from pathlib import Path

from fathom.models import (
    Contact,
    ContactSource,
    DetectionEvent,
    DetectionMethod,
    GramArtifact,
    GramType,
    Provenance,
    SourceModality,
)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def test_detection_event_roundtrip():
    event = DetectionEvent(
        correlation_id="abc",
        source_modality=SourceModality.ACOUSTIC,
        source_id="array-01:beam-04",
        timestamp=_now(),
        frequency_hz=7.4,
        snr_db=12.3,
        detection_method=DetectionMethod.CLASSICAL,
    )
    blob = event.model_dump_json()
    restored = DetectionEvent.model_validate_json(blob)
    assert restored == event


def test_contact_supports_multisource_provenance():
    now = _now()
    contact = Contact(
        contact_id="C-0001",
        sources=[
            ContactSource(modality=SourceModality.ACOUSTIC, source_id="array-01", last_seen=now),
            ContactSource(modality=SourceModality.AIS, source_id="ais-feed-1", last_seen=now),
        ],
        initiated_at=now,
        updated_at=now,
    )
    assert {s.modality for s in contact.sources} == {SourceModality.ACOUSTIC, SourceModality.AIS}


def test_gram_artifact_carries_provenance():
    prov = Provenance(
        timestamp=_now(),
        correlation_id="corr-1",
        parameter_snapshot={"n_fft": 16384},
    )
    artifact = GramArtifact(
        gram_type=GramType.LOFAR,
        image_path=Path("/tmp/x.png"),
        sidecar_path=Path("/tmp/x.png.audit.json"),
        provenance=prov,
    )
    blob = artifact.model_dump_json()
    restored = GramArtifact.model_validate_json(blob)
    assert restored.provenance.parameter_snapshot["n_fft"] == 16384