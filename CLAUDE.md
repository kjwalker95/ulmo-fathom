# CLAUDE.md — Ulmo / Fathom / Tuor Project Memory

Project memory for Claude Code sessions on the Ulmo build. Read at the start of every session.

## Project framing (PCD v3 three-layer split)

- **Ulmo** is the company (interim external posture: "Ulmo Defense" / "Ulmo Undersea" pending USPTO filing).
- **Fathom** is the platform: cloud-native, multi-source, ML-native, audit-tracked, classified-deployment-ready. Anduril Lattice analog. Not a standalone customer-purchasable product; ships embedded in whichever Ulmo product is consuming it.
- **Tuor** is Product 1: the modern IUSS-shore watch-station replacement built on Fathom. Anduril Sentry Tower analog. Currently in active build.
- Future Tolkien-Legendarium product names are committed to (next candidates evaluated at the Day-90 gate per PCD v3 §15.2: airborne ASW assistant + subsea cable protection).
- **Phase status:** Phase 0 (Weeks 1-6). Sprints 1 + 2 closed, Sprint 3 (Phase 0 exit) drafted.
- **Canonical docs:** `/Users/keith/Documents/Claude/Projects/Ulmo Product & Engineering/`
  - `PCD_v3.md` — product concept, source of truth. Three-layer Ulmo / Fathom / Tuor split. Supersedes PCD_v2.md and PCD_v1.md.
  - `UCD_v1.md` — company-level companion (identity, mission, capital strategy). Aligned with PCD at version boundaries.
  - `Phase0_Plan.md` — Phase 0 high-level plan (Weeks 1-6).
  - `Sprint{1,2}_Plan.md` + `Sprint{1,2}_Retro.md` — closed sprints.
  - `Sprint3_Plan.md` — Phase 0 exit sprint, drafted 2026-05-07.
  - `BrandDiligence_2026-05-05.md` — Ulmo + Fathom committed; Tuor diligence pending.
- **Repo:** `kjwalker95/ulmo-fathom` (GitHub). Local working tree at `/Users/keith/Documents/ulmo-fathom/`. Code under `src/fathom/` is logically a mix of Fathom platform substrate and Tuor product code (PCD v3 §6); physical reorg into `src/fathom/` + `src/tuor/` is a Phase 1 boundary refactor at earliest.
- **Data:** DeepShip at `/Users/keith/Documents/data/deepship/` (flat layout: `Cargo/103.wav`, etc. — each numeric .wav is a distinct vessel).

## User context

Keith Walker is Ulmo's CEO and sole founder (currently also CTO). Cleared. Four years on the IUSS watch floor in shore-based watch positions. Decade leading autonomous systems development at Sierra Space, Scout Space, and CACI; $30M+ captured contracts; $100M+ programs led (PCD v3 §5.6 — verification required before external use). Frame ASW / acoustic explanations against operator domain knowledge rather than from scratch — direct experience with LOFAR grams, blade-rate vulnerabilities (5-12 Hz tonals), auxiliary-machinery tonals (~50 Hz Russian-submarine signature), the line-of-interest / supervisor-escalation / senior-analyst-confirmation workflow, manual bearing intersection on the ICP, and the operator definition of "lost contact." Don't explain those concepts; reference them.

## Working style

- **Workflow: "act through me."** Keith executes commands and edits files himself rather than letting Claude Code run tools at scale. Claude produces runbook-style plans with copy-pasteable code blocks; Keith pastes, runs verification, commits. Hybrid mode: Claude executes pure boilerplate (scaffolding, env files, OpenAPI specs, configs, Dockerfile, doc updates) where there's no judgment; Keith executes substantive code (Pydantic models, ingestion, grams, detection algorithms, sanity scripts) where reading every line has value. Cluster-boundary checkpoints rather than per-file ones.
- **Plan files** live under `~/.claude/plans/`. Sprint 1: `users-keith-documents-claude-projects-u-rustling-kahan.md`. Sprint 2: `we-re-starting-sprint-2-sleepy-octopus.md`. Sprint 3 gets a fresh runbook plan when execution starts.
- **Commit cadence:** one commit per cluster, message format `sprint<N>: <cluster name>`. No `--no-verify` skips. Don't amend; create new commits.
- **Single-branch commit discipline (Sprint 2 retro lesson).** All commits land on `main` from the main repo at `/Users/keith/Documents/ulmo-fathom/`. The worktree at `.claude/worktrees/...` exists for Claude tool access; commits from there bifurcate the branch. If working from the worktree, sync with `git rebase main` before committing.
- **No GitHub PRs / no pushes** unless explicitly asked.

## Architectural binding decisions (PCD v3 + Sprint plans)

These bind every implementation decision; do not re-litigate without surfacing back to Keith.

- **LOFAR is linear-frequency, NOT mel-scale.** PCD v3 §6.6.
- **Calibrated uncertainty is first-order (platform-layer moat).** Deep ensemble + conformal prediction. PCD v3 §5.1; Phase 1 deliverable.
- **Vessel-level holdout for splits.** Recording-level splits leak. PCD v3 §12.2. The DeepShip published 98.58% likely uses recording-level splits; we don't compete with that number, we compete on equivalent splits.
- **Service-oriented internal structure from Day 1.** Even single-process Phase 0 separates modules with typed Pydantic interfaces. PCD v3 §7.1.
- **Platform vs Tuor capability split.** Five capabilities are Fathom platform substrate (ingestion, fusion, track maintenance, COP infrastructure, open API); six are Tuor product (LOFAR/DEMON, line detection on LOFAR, classification, multi-array bearing intersection, IUSS operator workflow, operator confirm/reject loop). PCD v3 §6. Code under `src/fathom/` reflects this split logically; physical reorg deferred.
- **Audit sidecars on every artifact.** Provenance is non-optional. PCD v3 §7.1.
- **Day-90 second-product candidate gate (PCD v3 §15.2).** End-of-Phase-2 decision: Path A (commit to one or both of airborne ASW assistant + subsea cable protection) or Path B (drop the platform claim, reposition as IUSS-modernization vendor). The platform claim's value depends on Path A landing.
- **No classified data in unclassified environments, ever.** `.gitignore` tripwire patterns are belt-and-suspenders; partition discipline is the real defense.

## Reference shortcuts for future sessions

- Sprint plan being executed → Sprint 3 plan at canonical-docs `Sprint3_Plan.md` (Phase 0 exit). Sprint 3 runbook gets a fresh `~/.claude/plans/` file when execution begins. Sprint 2 runbook: `~/.claude/plans/we-re-starting-sprint-2-sleepy-octopus.md`.
- DeepShip layout → flat (`<class>/<numeric_id>.wav`); the loader treats `recording_id` as `vessel_id` for vessel-level splits.
- Phase 0 exit gate → operator review of Tuor LOFAR grams + line-of-interest overlays at Sprint 3's tightened operating point + `Phase0_Review.md` drafted (Phase0_Plan.md §6).
- Sprint 1 substrate (platform-layer foundations):
  - `src/fathom/models.py` — typed contracts (Contact has multi-source provenance from Day 1; `LineOfInterest` defined for Sprint 2 to populate)
  - `src/fathom/events.py` — Topic enum + EventBus stub (Phase 1+ → Kafka/Redpanda)
  - `src/fathom/audit.py` — provenance & sidecar writer (every artifact carries one)
  - `src/fathom/grams/normalization.py:split_window_normalize` — single-pass; retained for display rendering continuity
- Sprint 2 substrate (Tuor classical line detection on platform TPSW):
  - `src/fathom/grams/normalization.py:tpsw_normalize` — platform-layer two-pass normalization; Tuor detection consumes this, display rendering still uses single-pass.
  - `src/fathom/detection/lines.py:detect_lines` — Tuor orchestrator; Phase 1 ML detection mirrors the same input/output shape (Topic.LINE_DETECTED schema is stable across classical → ML → fused, PCD v3 §6.7).
  - `src/fathom/detection/peaks.py` — both per-bin (Sprint 2 default) and 2D detectors; ablation note in `artifacts/sprint2_ablation/NOTES.md`.
  - `src/fathom/detection/persistence.py` — frequency-drift + gap-tolerance aware persistence filter.
  - `configs/sprint2.yaml` — frozen detection parameters; `far_sweep` block records the 3x3 grid characterized in `artifacts/sprint2_sanity/far_sweep.md`.
  - **Empirical finding:** default thresholds (peak_snr=8 dB, persistence=3 s) produce 67–1021 lines per DeepShip recording; even the tightest 3x3-sweep cell (10 dB, 5 s) yields ~2,033 lines/hour. Real ship audio has rich harmonic + cavitation structure that exceeds any fixed threshold; calibrated confidence (Phase 1; PCD v3 §5.1, platform-layer) is the path to operationally-reviewable rates.
- Sprint 3 substrate (Phase 0 exit deliverables):
  - `src/fathom/ingestion/_resample.py` — polyphase resampling primitive (52,734 Hz ShipsEar → 32,000 Hz target). Platform-layer.
  - `src/fathom/detection/merge.py` — post-hoc cluster-merge of nearby lines; coalesces STFT-leakage-split tonals into single representative lines. Tuor UX layer.
  - `src/fathom/models.py:SplitManifest` + `scripts/build_splits.py` — vessel-level train/val/test partitioner with SHA256 sidecar. Frozen splits Phase 1 reads from, never re-derives.
  - `scripts/sprint3_demo.py` — single-command Tuor demo. <1 s/recording on Apple Silicon; Phase 0 exit gate item #1.
  - `configs/sprint3.yaml` — `freq_min=3.0` (lifted from 1.0 Hz per CEO Phase 0 exit review to cut DC ramp while preserving slow-submarine blade-rate edge), tightened defaults (12 dB / 8 s + cluster-merge enabled), 5×5 `far_sweep` grid.
  - **Frozen Phase 1 evaluation baseline: (peak_snr=16 dB, persistence=20 s) → ~12 lines/hour.** Operationally tractable; ~1 line every 5 minutes. The bar Phase 1 ML+calibrated matches on throughput while adding per-line conformal coverage. Sprint 3 runtime default in `configs/sprint3.yaml` stays at (12, 8) for demos and classical-pipeline characterization; (16, 20) is evaluation anchor only.
- Smoke-test observations from Sprint 2 C1–C3 (TPSW behavior at vs near a tonal, 2D detector + gap_tolerance interaction, STFT leakage + drift_bins, synthetic FAR baseline) live in the Sprint 2 plan file under "Smoke-test observations (C1–C3, input to C6)."
- Phase 1 design memos → `docs/phase1_design/{A1_synthetic_generator.md, A2_ml_detection_architecture.md, A3_sim_to_real_evaluation.md}`. Binding specs for Sprint 4 implementation; signed off 2026-05-09 per `Design_Memo_Revision_Delta.md`. Key binding decisions: dual-architecture bakeoff (patch-CNN + U-Net), three-tier evaluation (synthetic exact-truth / real-ambient injection / operator-labeled panel), pre-computed KRAKEN/BELLHOP IRs (not runtime), frequency-axis heatmap localization (not single-line regression), truth manifest JSON sidecar on every synthetic clip (separate from audit/provenance sidecar).

## Anti-patterns to avoid

- Don't reproduce song lyrics or large copyrighted excerpts from source datasets.
- Don't propose using mel-scale, log-mel, or any perceptually-weighted frequency representation for operator-facing grams.
- Don't ship classified data or secret-tagged files. Don't `git add` files matching the tripwire patterns even if Keith requests it.
- Don't commit without an explicit Keith green light.
- Don't add framework dependencies (FastAPI, MLflow, Hydra, etc.) preemptively. Add when first needed.
- Don't conflate platform and product. "Fathom" refers to the platform substrate; "Tuor" refers to Product 1 (IUSS-shore watch station). Future products will earn their own Tolkien names; don't pre-claim them.
