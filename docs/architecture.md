# Fathom architecture (Sprint 1 substrate)

This document captures the platform-architectural commitments that bind from Day 1, even though most of them produce no visible Sprint 1 capability. Phase 0's job is to establish the substrate; Phase 1+ builds on it. The commitments derive from PCD v2 §7 ("Technical architecture") and Sprint1_Plan §3 ("Architectural decisions made in this sprint").

## Why these commitments matter on Day 1

The product Fathom is selling is not a watch-station tool. It is the modern command-and-control platform for undersea warfare — the 2020s software paradigm replacing the ICP's 1990s paradigm (PCD v2 §1, §2.2, §3.1). Every architectural decision in Sprint 1 either compounds toward that platform thesis or undercuts it. The commitments below are the ones that compound.

If we deferred any of them to Phase 1+, we would be retrofitting them under time pressure once the functional capability surface had grown. That fails. Architectural commitments are cheaper in Sprint 1 than in any subsequent sprint.

## Service-oriented internal structure

Modules under `src/fathom/` are separated as if they were independent services:
- `ingestion/` — accepts WAV/FLAC, indexes datasets, normalizes input.
- `grams/` — STFT, normalization, LOFAR, DEMON.
- `display/` — operator-friendly rendering.
- `models.py`, `events.py`, `audit.py` — shared cross-cutting modules.

Sprint 1 runs single-process. The structure supports Phase 1+ decomposition into separate services (Kubernetes pods, separate containers, separate deployment cadences) without rewrite. Each module has a typed Pydantic interface; consumers depend on the model, not on the implementation.

## Typed inter-module contracts

Pydantic v2 models in `src/fathom/models.py` are the typed contracts that cross every module boundary. They are also the source from which OpenAPI specs derive.

The `Contact` model supports multi-source provenance from Day 1 — `ContactSource` carries `SourceModality` (acoustic, AIS, SAR, sonobuoy, MAD), `source_id`, `last_seen`, and contributing `correlation_ids`. Sprint 1 only ingests acoustic data; AIS, SAR, sonobuoy, and MAD modalities slot in without rewrite (PCD v2 §6.5).

The `DetectionEvent` model carries `prediction_set` and `feature_attribution` fields that Sprint 1 leaves null. Phase 1's calibrated-uncertainty work (deep ensemble + conformal prediction; PCD v2 §5.1) populates them.

The `LineOfInterest` model is defined in Sprint 1 even though Sprint 2 builds the detection logic. Defining the contract early lets `events.py` and `audit.py` reach a stable shape that Sprint 2 doesn't have to break.

## OpenAPI specs scaffolded

Specs live under `apis/` and are validated against OpenAPI 3.1 in CI (Phase 1+) and on-demand via `openapi-spec-validator` locally. Sprint 1 has no running HTTP server — the specs document the contract that the Pydantic models implement.

Phase 1+ external API surface (PCD v2 §6.9) derives from these specs. The internal services that come up in Phase 1 will consume the same APIs external partners will consume — no second-class internal-only interface.

## Structured event emission

Every module emits well-formed events to an in-memory pub/sub bus (`src/fathom/events.py`). Topics are defined in the `Topic` enum:
- `gram.generated` — landed in Sprint 1.
- `line.detected` — Sprint 2.
- `contact.initiated`, `contact.updated` — Phase 2.

Phase 1+ swaps the in-memory bus for Kafka or Redpanda. The `Topic` enum and the Pydantic payload schemas stay stable across the swap.

## Provenance and audit from Day 1

Every artifact written by Fathom carries a JSON sidecar with full provenance (`src/fathom/audit.py`):
- `timestamp` (UTC, ISO 8601)
- `correlation_id` (UUID4)
- `source_recording_path`
- `dataset_manifest_hash` (SHA256 over the manifest JSON)
- `code_commit_hash` (git rev-parse HEAD)
- `parameter_snapshot` (every config used to produce the artifact)

Sprint 1 writes per-gram audit sidecars. The same utility powers per-classification audit in Phase 1, per-track audit in Phase 2, and per-API-call audit when the external surface lands. Operators and program offices consuming Fathom outputs can trace any decision back to the inputs and code that produced it.

This is the foundation of the per-decision audit-trail commitment in PCD v2 §7.1 and Section 12.6.

## Methodological commitments encoded as code

- **Vessel-level metadata preserved.** Every loaded recording carries `vessel_id`, `recording_id`, `dataset`, `sample_rate_hz`, and `duration_s`. Phase 1 splits depend on it. The DeepShip release on disk uses a flat `<class>/<numeric_id>.wav` layout where each numeric ID is a distinct vessel; the loader treats `recording_id` as `vessel_id` for that release.
- **Manifest hashes mandatory.** Every dataset index produces a JSON manifest with a SHA256 sidecar so any future training run can prove what data it consumed.
- **Sensor-agnostic ingestion.** The ingestion layer accepts WAV/FLAC in Sprint 1; the API is shaped so adding streaming sources, sonobuoy data link parsers, or AIS feeds later does not require changing downstream consumers.
- **Mono reduction with caveat.** v1 Sprint 1 reduces multi-channel waveforms to mono (omnidirectional pressure) for gram generation. Bearing channels are preserved in metadata for Phase 2 use.
- **Classified-tag tripwire.** `.gitignore` patterns for `*classified*`, `*CLASSIFIED*`, `*SECRET*`, `*confidential*`, etc. Belt-and-suspenders only; partition discipline is the real defense.

## What is explicitly deferred

Sprint 1 does not ship: ML model code, calibrated uncertainty, line detection, classification, tracking, bearing intersection, COP display, multi-sensor handling, AIS ingestion, multi-source fusion, streaming ingestion at production scale, external API exposure, cloud deployment, sonobuoy data link parsing, or 3D tactical visualization. Each lands in the phase indicated in `Phase0_Plan.md` and the subsequent phase plans.

The architectural substrate above is what makes those deferrals safe. We add capability without rewriting.
