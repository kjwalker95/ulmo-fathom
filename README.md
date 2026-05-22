
# Ulmo · Fathom · Tuor

**Ulmo** is the company. **Fathom** is the platform. **Tuor** is Product 1.

This repository is the engineering home of the Ulmo build. **Fathom** is the cloud-native, multi-source, ML-native, audit-tracked, classified-deployment-ready platform for undersea warfare. **Tuor** is the first product on Fathom: the modern IUSS-shore watch-station replacement that ingests passive acoustic data, generates LOFAR/DEMON grams, automatically detects lines of interest, classifies contacts against the signature library with calibrated uncertainty, maintains tracks, and computes bearing intersections (Anduril Sentry Tower analog). Tuor replaces the human-analyst workflow currently running on Lockheed Martin's Integrated Common Processor — a 1990s software paradigm running 21st-century missions — with a 2020s software paradigm built around how IUSS operators actually discriminate contacts.

Phase 0 (Weeks 1-6) is **architecture validation**. Three sprints closed:
- **Sprint 1** — repo, ingestion, LOFAR/DEMON grams, platform foundations (typed Pydantic contracts, in-memory event-bus stub, audit sidecars on every artifact, OpenAPI specs scaffolded, containerized).
- **Sprint 2** — Tuor classical line detection (TPSW two-pass + persistence-filtered peak detection + line-of-interest reporting) on Fathom platform substrate.
- **Sprint 3** — full-dataset characterization, ShipsEar resampling, post-hoc cluster-merge, vessel-level split manifests, single-command demo, Phase 0 exit review.

Phase 1 (Weeks 7-15) is the **calibrated-uncertainty phase**. Two sprints closed:
- **Sprint 4** — synthetic LOFAR data generator (parameterized tonals, biological confusers, C1.3-lite parametric three-path channel + Thorpe absorption), ML detection models (ResNet-18 patch-CNN + U-Net + clDice), Tier-1 evaluation harness, classical-vs-ML smoke. Established: dense mask supervision outperforms sparse heatmap at small-data scale (Sprint 4 retro).
- **Sprint 5** — mix-and-train substrate, Tier-2 real-ambient injection evaluation harness, 21-cell ratio sweep on cloud GPU, multi-threshold aggregation, single-model calibration baseline (with bimodal-saturation finding), pretrain-finetune ablation (clean negative), operator-recognition test on real DeepShip recordings (PCD v4 §13.1 PASS). Phase 1 exit gate (PCD v4 §15.2) reads PASS on Gates 1 + 3, baseline on Gate 2.

Product framing: see `PCD_v4.md` (canonical; supersedes PCD v3) and Phase plans (`Phase0_Plan.md`, `Phase1_Plan.md`) in the engineering project root. Per-sprint plans + retros: `Sprint{1..5}_Plan.md` and `Sprint{1..5}_Retro.md`. Architectural commitments inside this repo: `docs/architecture.md` + design memos under `docs/phase1_design/`.
>>>>>>> e635e76 (sprint5: Z — close-out (CLAUDE.md + README PCD v4 alignment))

## Quickstart

Requires Python 3.11+ on `PATH` (Homebrew `brew install python@3.11` is the easiest install on macOS).

```bash
# Create and activate a virtualenv
python3.11 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev tooling and PyTorch (Phase 1 readiness)
pip install --upgrade pip
pip install -e ".[dev,torch]"

# Smoke-test
pytest -v

# Single-command Tuor demo on one recording: WAV in -> LOFAR + detected lines out
python scripts/sprint3_demo.py /path/to/recording.wav
open artifacts/sprint3_demo/<recording>.png

# Full-dataset Tuor sanity check across DeepShip
python scripts/sanity_check_lines.py \
    --config configs/sprint3.yaml \
    --deepship-root /path/to/deepship \
    --out-dir artifacts/sprint3_sanity \
    --n-per-class 1000
open artifacts/sprint3_sanity/INDEX.md

# 5x5 FAR sweep (peak SNR x persistence)
python scripts/far_sweep.py \
    --config configs/sprint3.yaml \
    --deepship-root /path/to/deepship \
    --n-per-class 1000

# Vessel-level train/val/test splits (frozen via SHA256 sidecar; Phase 1 reads from these)
python scripts/build_splits.py --deepship-root /path/to/deepship
```

PyTorch installs the CPU/MPS wheel automatically on Apple Silicon. For Linux x86_64 + CUDA 12.1, install with `pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu121` then add `torch torchaudio`.

## Container usage (verified Sprint 3)

```bash
docker build -t fathom:sprint3 .

docker run --rm \
    -v /path/to/deepship:/data/deepship:ro \
    -v $(pwd)/artifacts:/app/artifacts \
    fathom:sprint3 \
    --config configs/sprint3.yaml \
    --deepship-root /data/deepship \
    --out-dir artifacts/sprint3_container \
    --n-per-class 1
```

Container output matches host output within float-precision tolerance modulo timestamps, bind-mount paths, and `correlation_id` UUIDs. Specifically:

- **Manifest content hash** (path and `index_built_at` fields stripped, then SHA256 over canonical JSON) is bit-identical between host and container.
- **Substantive line-of-interest fields** — `frequency_hz`, `bandwidth_hz`, `persistence_s` — are bit-identical (derived from STFT bin indexing math).
- **`snr_db`** differs by at most ~10⁻⁵ dB across BLAS / libm implementations on the same architecture (macOS libSystem vs Linux musl on ARM64). This is operationally invisible. The 1e-3 tolerance used in the parity check is 100× the observed drift.

The Sprint 1 README's claim that the file-level manifest SHA matched host vs container was incorrect; it never did, because `root` and per-recording `path` fields contain bind-mount paths that legitimately differ inside the container. The corrected parity definition above is content-based.

If `docker` is missing from `PATH` on macOS even though Docker Desktop is installed, the binary lives at `/Applications/Docker.app/Contents/Resources/bin/docker` — add that directory to `PATH` in `~/.zshrc`.

## Repository layout

```
fathom/
├── README.md                       project framing, quickstart
├── CLAUDE.md                       project memory for Claude Code sessions
├── pyproject.toml                  package metadata + deps, ruff/black/mypy/pytest config
├── Dockerfile                      Phase 0 demo containerization
├── apis/                           OpenAPI 3.1 specs (internal in Phase 0)
│   ├── ingestion.openapi.yaml      Fathom platform ingestion (PCD v4 §6.1)
│   ├── grams.openapi.yaml          Tuor gram generation (PCD v4 §6.6)
│   └── detection.openapi.yaml      Tuor line detection (PCD v4 §6.7)
├── configs/                        per-sprint frozen parameters
│   ├── sprint1.yaml                LOFAR/DEMON/ingestion/display defaults
│   ├── sprint2.yaml                + classical detection (snr=8 dB, persistence=3 s)
│   └── sprint3.yaml                + cluster-merge, tightened defaults (snr=12 dB, persistence=8 s),
│                                     5x5 far_sweep grid
├── docs/                           architecture, data access, ICP display conventions
├── src/fathom/                     Python package
│   ├── models.py                   Pydantic contracts (Contact, DetectionEvent,
│   │                               LineOfInterest, SplitManifest, ...)
│   ├── events.py                   In-memory pub/sub event-bus stub (Phase 1+ -> Kafka/Redpanda)
│   ├── audit.py                    Provenance & audit-sidecar utilities
│   ├── ingestion/                  DeepShip + ShipsEar loaders + polyphase resampling
│   ├── grams/                      LOFAR + DEMON gram generation, single-pass + TPSW normalization
│   ├── detection/                  Classical line detection (peaks, persistence, lines, merge)
│   └── display/                    Operator-friendly gram rendering with overlay support
├── scripts/
│   ├── sanity_check_grams.py       Sprint 1 gram sanity check
│   ├── sanity_check_lines.py       Sprint 2+ detection sanity check
│   ├── far_sweep.py                Sprint 2+ N×N (peak_snr, persistence) FAR characterization
│   ├── build_splits.py             Sprint 3 vessel-level train/val/test partitioner
│   └── sprint3_demo.py             Single-command Tuor demo (Phase 0 exit deliverable)
├── tests/                          pytest unit + smoke tests
└── artifacts/                      gitignored experiment outputs
```

## Architectural commitments (from PCD v4)

- **Three-layer naming.** Ulmo (company) / Fathom (platform) / Tuor (Product 1). Future products earn their own Tolkien-Legendarium names; don't pre-claim.
- **Service-oriented internal structure.** Modules under `src/fathom/` separated as if they were independent services with typed Pydantic contracts. Phase 1+ decomposes module-shaped services into process-shaped containers without rewrite.
- **OpenAPI specs scaffolded.** Each module's external interface specified in OpenAPI YAML; Phase 1+ external API surface derives from these specs.
- **Structured event emission.** Every module emits well-formed events to an in-memory pub/sub bus. Phase 1+ swaps the in-memory bus for Kafka or Redpanda; the schema stays.
- **Provenance and audit from Day 1.** Every artifact written carries a JSON sidecar with full provenance: timestamp, correlation ID, source recording path, dataset manifest hash, code commit hash, parameter snapshot.
<<<<<<< HEAD
- **Calibrated uncertainty is first-order (platform moat).** Deep ensemble + conformal prediction in Phase 1 (PCD v3 §5.1). All Tuor and future-product classifiers consume the same calibration architecture.

=======
- **Calibrated uncertainty is first-order (platform moat).** Deep ensemble + conformal prediction in Phase 1 (PCD v4 §5.1). All Tuor and future-product classifiers consume the same calibration architecture.
- **Phase 1 exit technical validation gate (PCD v4 §15.2 + v4.1).** Platform play is committed. Three sub-gates: classification on real ambient data ("better than nothing and improving with data"); calibrated uncertainty (deep ensemble + conformal, ECE < 0.05, coverage tracks alpha); platform composability (second pipeline consumes Fathom services without modification). Leading Product 2 candidate per v4.1: USW DSS (AN/UYQ-100) displacement on surface combatants (DDG/CVN install base, FY27 MAC IDIQ vehicle). Airborne ASW second; cable protection opportunistic only.
>>>>>>> e635e76 (sprint5: Z — close-out (CLAUDE.md + README PCD v4 alignment))

## Methodological commitments

- **LOFAR is linear-frequency.** Mel-scale perceptually weights frequencies for human auditory perception, which is the wrong choice for operator-facing analysis of low-frequency tonal lines.
- **Vessel-level metadata preserved through ingestion.** Phase 1's training/eval splits depend on it. Recording-level splits leak; published DeepShip numbers using them are inflated and not the bar we measure against.
- **Vessel-level holdout enforced via `SplitManifest`.** Splits frozen via SHA256 sidecar; downstream code reads from the manifest, never re-derives.
- **Manifest hashes are mandatory.** Every dataset index produces a JSON manifest with a SHA256 sidecar so any future training run can prove what data it consumed.
- **Audit sidecars are mandatory.** Every gram, every detection event, every classification decision carries a JSON sidecar with full provenance.
- **No classified data.** Partition discipline from the first commit. `.gitignore` carries belt-and-suspenders tripwire patterns.

## License

Proprietary. All rights reserved.
