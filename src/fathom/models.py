"""Pydantic data models for Fathom platform entities.

These models are the typed inter-module contracts (Sprint1_Plan §3). They are also
the source from which OpenAPI specs derive. The Contact model supports multi-source
provenance from Day 1 even though Sprint 1 only ingests acoustic data; AIS, SAR,
sonobuoy, and MAD modalities slot in without rewrite (PCD v2 §6.5).
"""
from __future__ import annotations

import enum
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceModality(str, enum.Enum):
    """Sensor modality for a detection source. Acoustic only in Sprint 1."""
    ACOUSTIC = "acoustic"
    AIS = "ais"
    SAR = "sar"
    SONOBUOY = "sonobuoy"
    MAD = "mad"


class GramType(str, enum.Enum):
    LOFAR = "lofar"
    DEMON = "demon"


class DetectionMethod(str, enum.Enum):
    CLASSICAL = "classical"
    ML = "ml"
    FUSED = "fused"


class ClassificationLevel1(str, enum.Enum):
    BIOLOGICAL = "biological"
    ENVIRONMENTAL = "environmental"
    SURFACE_VESSEL = "surface_vessel"
    SUBMERGED_VESSEL = "submerged_vessel"
    UNKNOWN = "unknown"


class ClassificationLevel2(str, enum.Enum):
    MERCHANT_CARGO = "merchant_cargo"
    MERCHANT_TANKER = "merchant_tanker"
    MERCHANT_CONTAINER = "merchant_container"
    FISHING = "fishing"
    MILITARY_SURFACE = "military_surface"
    SUBMARINE = "submarine"
    UUV = "uuv"
    UNKNOWN = "unknown"


class StftConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    sample_rate: int
    n_fft: int
    hop_length: int
    window_length: int
    window: str = "hanning"


class LOFARConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    stft: StftConfig
    freq_min_hz: float
    freq_max_hz: float
    log_epsilon: float = 1.0e-10
    normalization_train_window_bins: int
    normalization_central_window_bins: int
    normalization_gap_bins: int


class DEMONConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    sample_rate: int
    band_low_hz: float
    band_high_hz: float
    envelope_lpf_cutoff_hz: float
    decimation_factor: int
    n_fft: int
    hop_length: int
    window: str = "hanning"


class RecordingMetadata(BaseModel):
    """Vessel-level metadata enforced through ingestion (Sprint1_Plan §3)."""
    recording_id: str
    vessel_id: str | None
    dataset: str
    class_label: str | None
    sample_rate_hz: int
    duration_s: float
    n_channels: int
    path: Path
    notes: str | None = None


class DatasetIndex(BaseModel):
    """An indexed dataset. Used as input to gram generation and (later) training."""
    dataset: str
    root: Path
    recordings: list[RecordingMetadata]
    index_built_at: datetime


class Provenance(BaseModel):
    """Audit-trail fields carried by every produced artifact (Sprint1_Plan §3)."""
    timestamp: datetime
    correlation_id: str
    source_recording_path: Path | None = None
    dataset_manifest_hash: str | None = None
    code_commit_hash: str | None = None
    parameter_snapshot: dict[str, Any] = Field(default_factory=dict)


class GramArtifact(BaseModel):
    """Reference to a produced LOFAR or DEMON gram artifact on disk."""
    gram_type: GramType
    image_path: Path
    sidecar_path: Path
    provenance: Provenance
    duration_s: float | None = None


class DetectionEvent(BaseModel):
    """Single per-source detection event. Multiple events with consistent
    space-time-signature fuse into a Contact (PCD v2 §6.5)."""
    correlation_id: str
    source_modality: SourceModality
    source_id: str
    timestamp: datetime
    frequency_hz: float | None = None
    bandwidth_hz: float | None = None
    bearing_deg: float | None = None
    snr_db: float | None = None
    confidence: float | None = None
    prediction_set: list[str] | None = None
    detection_method: DetectionMethod | None = None
    feature_attribution: dict[str, float] | None = None


class ContactSource(BaseModel):
    """Per-source presence inside a multi-source Contact."""
    modality: SourceModality
    source_id: str
    last_seen: datetime
    contributing_events: list[str] = Field(default_factory=list)


class Contact(BaseModel):
    """A platform-level contact entity. Supports multi-source provenance from Day 1."""
    contact_id: str
    classification_level1: ClassificationLevel1 | None = None
    classification_level2: ClassificationLevel2 | None = None
    classification_level3: str | None = None
    fused_confidence: float | None = None
    sources: list[ContactSource] = Field(default_factory=list)
    initiated_at: datetime
    updated_at: datetime
    status: str = "active"


class LineOfInterest(BaseModel):
    """Operational line-of-interest report (PCD v2 §6.3). Sprint 2 deliverable;
    the model is defined now so events.py and audit.py have a stable contract."""
    correlation_id: str
    array_id: str
    beam_id: str | None = None
    timestamp: datetime
    frequency_hz: float
    bandwidth_hz: float | None = None
    snr_db: float
    persistence_s: float
    detection_method: DetectionMethod
    confidence: float | None = None