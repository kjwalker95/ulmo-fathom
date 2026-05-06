# Fathom

The modern command-and-control platform for undersea warfare.

This repository is the engineering home of Fathom — Ulmo's first product and the platform on which the rest of Ulmo's portfolio is built. Fathom ingests passive acoustic data from undersea arrays, fuses it with non-acoustic sources (AIS in v1, satellite SAR in v1.5, sonobuoy and MAD when classified data lands), runs ML-augmented detection and classification with calibrated uncertainty against a proprietary signature library, maintains tracks over time, and exposes every capability via open API. The platform replaces the human-analyst detection-classification-fusion-display workflow currently running on Lockheed Martin's Integrated Common Processor (ICP) — a 1990s software paradigm running 21st-century missions — with a 2020s software paradigm built around how IUSS operators actually discriminate contacts.

Phase 0 (Weeks 1-6) is **architecture validation**. Sprint 1 (Weeks 1-2) stands up the repository, the WAV → LOFAR-gram pipeline, and the platform-architectural foundations (typed inter-module contracts, in-memory event-bus stub, audit sidecars on every artifact, OpenAPI specs scaffolded, Docker container) on which Sprint 2 builds classical line detection.

Product framing: see `PCD_v2.md` (canonical) and `Phase0_Plan.md` in the engineering project root. Sprint-level scope: `Sprint1_Plan.md`. Architectural commitments inside this repo: `docs/architecture.md`.

## Quickstart

```bash
# Create the conda environment (macOS Apple Silicon defaults; Linux/CUDA swap noted in environment.yml)
conda env create -f environment.yml
conda activate fathom

# Smoke-test
pytest -v

# Generate sanity-check artifacts from DeepShip
python scripts/sanity_check_grams.py \
    --deepship-root /path/to/deepship \
    --out-dir artifacts/sprint1_sanity \
    --n-per-class 5
open artifacts/sprint1_sanity/INDEX.md
```

## Repository layout

```
fathom/
├── README.md                       project framing, quickstart
├── CLAUDE.md                       project memory for Claude Code sessions
├── environment.yml                 conda env (Apple Silicon defaults)
├── pyproject.toml                  package metadata, ruff/black/mypy/pytest config
├── Dockerfile                      Phase 0 demo containerization
├── apis/                           OpenAPI 3.1 specs (internal in Sprint 1)
├── configs/                        YAML config (sprint1.yaml)
├── docs/                           architecture, data access, ICP display conventions
├── src/fathom/                     Python package
│   ├── models.py                   Pydantic data models (Contact, DetectionEvent, ...)
│   ├── events.py                   In-memory pub/sub event-bus stub
│   ├── audit.py                    Provenance & audit-sidecar utilities
│   ├── ingestion/                  DeepShip + ShipsEar loaders
│   ├── grams/                      LOFAR + DEMON gram generation, normalization
│   └── display/                    Operator-friendly gram rendering
├── scripts/sanity_check_grams.py   Sprint 1 sanity-check CLI (operator review deliverable)
├── tests/                          pytest smoke tests
└── artifacts/                      gitignored experiment outputs
```

## Architectural commitments

The platform reframe (PCD v2) introduces architectural commitments that bind from Day 1 even though most of them produce no visible Sprint 1 capability:

- **Service-oriented internal structure.** Modules separated as if they were independent services, with typed Pydantic interfaces between them.
- **OpenAPI specs scaffolded.** Each module's external interface specified in OpenAPI YAML; Phase 1+ external API surface derives from these specs.
- **Structured event emission.** Every module emits well-formed events to an in-memory pub/sub bus. Phase 1+ swaps the in-memory bus for Kafka or Redpanda; the schema stays.
- **Provenance and audit from Day 1.** Every artifact written carries a JSON sidecar with full provenance: timestamp, correlation ID, source recording path, dataset manifest hash, code commit hash, parameter snapshot.

See `docs/architecture.md` for rationale.

## Methodological commitments

- **LOFAR is linear-frequency.** Mel-scale perceptually weights frequencies for human auditory perception, which is the wrong choice for operator-facing analysis of low-frequency tonal lines.
- **Vessel-level metadata is preserved through ingestion.** Phase 1's training/eval splits depend on it. Recording-level splits leak; published DeepShip numbers using them are inflated and not the bar we measure against.
- **Manifest hashes are mandatory.** Every dataset index produces a JSON manifest with a SHA256 sidecar so any future training run can prove what data it consumed.
- **Audit sidecars are mandatory.** Every gram, every detection event, every classification decision carries a JSON sidecar with full provenance.
- **No classified data, ever.** Partition discipline from the first commit. `.gitignore` carries belt-and-suspenders tripwire patterns.

## License

Proprietary. All rights reserved.
