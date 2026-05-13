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
    """Operational line-of-interest report (PCD v3 §6.7). Tuor product capability;
    schema lives at the platform layer because Topic.LINE_DETECTED is platform
    infrastructure stable across classical -> ML -> fused detection methods."""
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


class SplitManifest(BaseModel):
    """Vessel-level train/val/test split manifest (Sprint 3 Cluster 3).

    Built once via `scripts/build_splits.py` and frozen via SHA256 sidecar.
    Phase 1 training joins manifest vessel IDs to the dataset index; the manifest
    is never re-derived from raw data, ensuring reproducibility across model runs.

    Per CLAUDE.md / PCD v3 §12.2 architectural binding: vessel-level holdout is
    enforced; recording-level splits leak. Platform-layer infrastructure (data-
    management plumbing reusable across all Fathom products).

    Vessel ID contract (Sprint 5 A0 fix, 2026-05-13): entries in `train_vessels` /
    `val_vessels` / `test_vessels` are compound keys of the form
    `<class>/<recording_id>` (e.g., `"Cargo/103"`, `"Tanker/41"`). This
    disambiguates DeepShip's flat-layout numeric-ID collisions across class
    folders. Downstream consumers parse via `class_label, stem = key.split("/", 1)`.
    Pre-A0 manifests stored bare numeric IDs and are not interpretable without
    out-of-band class lookup; regenerate before use.
    """
    dataset: str
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    stratified_by_class: bool
    train_vessels: list[str]
    val_vessels: list[str]
    test_vessels: list[str]
    built_at: datetime
    notes: str | None = None

class SyntheticPropagationGeometry(BaseModel):
    """Per-source propagation geometry for C1.3-lite three-path channel.

    Pydantic mirror of `fathom.synthetic.priors.SampledPropagationGeometry`
    used for truth-manifest serialization. The sampling-time dataclass and
    this serialization model share the same field shape; conversion is
    one-to-one.
    """
    water_depth_m: float
    source_depth_m: float
    receiver_depth_m: float
    horizontal_range_m: float
    sound_speed_m_per_s: float
    bottom_reflection_loss_db: float


class SyntheticLineGroundTruth(BaseModel):
    """Per-line ground truth for a synthetic LOFAR clip (A1 §3.3.1)."""
    line_id: str
    source_id: str | None = None
    source_type: str = "tonal"      # future: "biological", "broadband"
    harmonic_id: int = 0            # 0 = fundamental
    f0_hz: float
    freq_curve_hz: list[float]      # frequency at each STFT frame
    t_start_s: float
    t_end_s: float
    snr_curve_db: list[float]       # per-frame SNR
    persistence_s: float
    drift_rate_hz_per_s: float = 0.0
    mask_bin_indices: list[tuple[int, int]] = Field(default_factory=list)
    generation_seed: int
    propagation_geometry: SyntheticPropagationGeometry | None = None
    propagation_model_id: str | None = None  # e.g. "c1_3_lite_three_path_v1"


class SyntheticConfuserLabel(BaseModel):
    """Biological confuser metadata (populated in C1 when biologicals land)."""
    species: str
    species_code: str | None = None
    confuser_clip_id: str
    source_dataset: str | None = None
    t_start_s: float
    t_end_s: float
    freq_range_hz: tuple[float, float]
    target_snr_db: float | None = None


class SyntheticTruthManifest(BaseModel):
    """Synthetic clip-level truth manifest (A1 §3.3.1).

    Separate from the audit/provenance sidecar — provenance tracks how the clip
    was produced; this manifest tracks what's IN the clip as ground truth for
    A2 training and A3 evaluation.
    """
    clip_id: str
    lines: list[SyntheticLineGroundTruth] = Field(default_factory=list)
    negative_label: bool = False
    confuser_labels: list[SyntheticConfuserLabel] = Field(default_factory=list)
    ambient_source_id: str
    ambient_source_clip_timestamp: str | None = None
    propagation_environment_id: str | None = None  # null in B1; set in C1
    generator_version: str

class BiologicalClip(BaseModel):
    """Individual clip in a biological-confuser library.

    Source-agnostic schema. C1.2.a's DCLDE 2018 extraction populates this;
    future Watkins/other extractions will produce the same shape.
    """
    clip_id: str
    source_dataset: str            # "dclde_2018"; future "watkins", etc.
    species_code: str              # "Bm" (blue whale), "Eg" (right whale)
    species_name: str              # "blue_whale", "north_atlantic_right_whale"
    site: str
    deployment: str
    sample_rate_hz: int            # native (e.g., 2000 for DCLDE LF)
    duration_s: float              # full clip incl. pad
    pad_s: float                   # pre + post each side
    annotated_t_start_s: float     # annotation onset within clip (= pad_s)
    annotated_t_end_s: float       # annotation offset within clip
    freq_range_hz: tuple[float, float]
    quality: str
    sha256: str
    relative_path: str             # relative to library root


class BiologicalClipLibrary(BaseModel):
    """Directory-of-WAVs + manifest. Loaded by fathom.synthetic.biologicals;
    library is source-agnostic — different `source_dataset` values OK."""
    library_id: str                # "dclde_2018_lf_v1"
    source_dataset: str
    n_clips: int
    species_counts: dict[str, int]
    clips: list[BiologicalClip]
    built_at: datetime