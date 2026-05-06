# CLAUDE.md — Fathom Project Memory

Project memory for Claude Code sessions on the Fathom build. Read at the start of every session.

## Project framing

- **Product:** Fathom — the modern command-and-control platform for undersea warfare. Phase 0 (Weeks 1-6); Sprint 1 closed, Sprint 2 (classical line detection) in progress.
- **Company:** Ulmo (interim "Ulmo Defense" / "Ulmo Undersea" external posture pending USPTO filing).
- **Canonical docs:** `/Users/keith/Documents/Claude/Projects/Ulmo Product & Engineering/`
  - `PCD_v2.md` — product concept, source of truth (supersedes PCD_v1.md)
  - `Phase0_Plan.md` — Phase 0 high-level plan (Weeks 1-6)
  - `Sprint1_Plan.md` — Sprint 1 detailed plan (Weeks 1-2)
  - `BrandDiligence_2026-05-05.md` — brand decision record
  - Sprint2_Plan.md, Sprint3_Plan.md drafted at end of each predecessor
- **Repo:** `kjwalker95/ulmo-fathom` (GitHub). Local working tree at `/Users/keith/Documents/ulmo-fathom/`.
- **Data:** DeepShip at `/Users/keith/Documents/data/deepship/` (flat layout: `Cargo/103.wav`, etc. — each numeric .wav is a distinct vessel).

## User context

Keith Walker is Ulmo's CEO and (currently) CTO. Cleared. Four years on the IUSS watch floor in shore-based watch positions. Frame ASW / acoustic explanations against operator domain knowledge rather than from scratch — he has direct experience with LOFAR grams, blade-rate vulnerabilities (5-12 Hz tonals), auxiliary-machinery tonals (~50 Hz Russian-submarine signature), the line-of-interest / supervisor-escalation / senior-analyst-confirmation workflow, manual bearing intersection on the ICP, and the operator definition of "lost contact." Don't explain those concepts; reference them.

## Working style

- **Workflow: "act through me."** For this project Keith executes commands and edits files himself rather than letting Claude Code run tools at scale. Claude produces runbook-style plans with copy-pasteable code blocks; Keith pastes, runs verification, commits. Hybrid mode is acceptable: Claude executes pure boilerplate (scaffolding, env files, OpenAPI specs, configs, Dockerfile) where there's no judgment; Keith executes substantive code (Pydantic models, ingestion, grams, sanity scripts) where reading every line has value. Cluster-boundary checkpoints rather than per-file ones.
- **Plan files** live under `~/.claude/plans/`. Sprint 1 runbook: `users-keith-documents-claude-projects-u-rustling-kahan.md`. Sprint 2 runbook: `we-re-starting-sprint-2-sleepy-octopus.md`.
- **Commit cadence:** one commit per cluster, message format `sprint<N>: <cluster name>`. No `--no-verify` skips. Don't amend; create new commits.
- **No GitHub PRs / no pushes** unless explicitly asked.

## Architectural binding decisions (from PCD v2 and Sprint1_Plan §3)

These bind every implementation decision; do not re-litigate without surfacing back to Keith.

- **LOFAR is linear-frequency, NOT mel-scale.** PCD v2 §6.2.
- **Calibrated uncertainty is first-order.** Deep ensemble + conformal prediction. Phase 1 deliverable; not Sprint 1.
- **Vessel-level holdout for splits.** Recording-level splits leak. The DeepShip published 98.58% likely uses recording-level splits; we don't compete with that number, we compete on equivalent splits.
- **Service-oriented internal structure from Day 1.** Even single-process Sprint 1 separates modules with typed Pydantic interfaces.
- **Audit sidecars on every artifact.** Provenance is non-optional.
- **No classified data in unclassified environments, ever.** `.gitignore` tripwire patterns are belt-and-suspenders; partition discipline is the real defense.

## Reference shortcuts for future sessions

- Sprint plan being executed → `~/.claude/plans/we-re-starting-sprint-2-sleepy-octopus.md` (Sprint 2)
- DeepShip layout → flat (`<class>/<numeric_id>.wav`); the loader treats `recording_id` as `vessel_id` for vessel-level splits
- Phase 0 exit gate → operator review of LOFAR grams + line-of-interest overlays (Keith eyeballs `artifacts/sprint2_sanity/INDEX.md` against ICP intuition)
- Sprint 1 substrate that Sprint 2+ builds on:
  - `src/fathom/models.py` — typed contracts (Contact has multi-source provenance from Day 1; `LineOfInterest` defined in Sprint 1 for Sprint 2 to populate)
  - `src/fathom/events.py` — Topic enum + EventBus stub (Phase 1+ → Kafka/Redpanda)
  - `src/fathom/audit.py` — provenance & sidecar writer (every artifact carries one)
  - `src/fathom/grams/normalization.py:split_window_normalize` — single-pass; retained for display rendering continuity
- Sprint 2 substrate that Sprint 3+ builds on:
  - `src/fathom/grams/normalization.py:tpsw_normalize` — two-pass; detection consumes this, display rendering still uses single-pass for Sprint 1 continuity
  - `src/fathom/detection/lines.py:detect_lines` — top-level orchestrator; Phase 1 ML detection mirrors the same input/output shape (Topic.LINE_DETECTED schema is stable across classical → ML → fused)
  - `src/fathom/detection/peaks.py` — both per-bin (Sprint 2 default) and 2D detectors; ablation note in `artifacts/sprint2_ablation/NOTES.md`
  - `src/fathom/detection/persistence.py` — frequency-drift + gap-tolerance aware persistence filter
  - `configs/sprint2.yaml` — frozen detection parameters; `far_sweep` block records the 3x3 grid characterized in `artifacts/sprint2_sanity/far_sweep.md`
  - **Empirical finding:** default thresholds (peak_snr=8 dB, persistence=3 s) produce 67–1021 lines per DeepShip recording; even the tightest 3x3-sweep cell (10 dB, 5 s) yields ~2,033 lines/hour. Real ship audio has rich harmonic + cavitation structure that exceeds any fixed threshold; calibrated confidence (Phase 1; PCD v2 §5.1) is the path to operationally-reviewable rates.
- Smoke-test observations from Sprint 2 C1–C3 (TPSW behavior at vs near a tonal, 2D detector + gap_tolerance interaction, STFT leakage + drift_bins, synthetic FAR baseline) live in the Sprint 2 plan file under "Smoke-test observations (C1–C3, input to C6)."

## Anti-patterns to avoid

- Don't reproduce song lyrics or large copyrighted excerpts from source datasets.
- Don't propose using mel-scale, log-mel, or any perceptually-weighted frequency representation for operator-facing grams.
- Don't ship classified data or secret-tagged files. Don't `git add` files matching the tripwire patterns even if Keith requests it.
- Don't commit without an explicit Keith green light.
- Don't add framework dependencies (FastAPI, MLflow, Hydra, etc.) preemptively. Add when first needed.
