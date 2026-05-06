# CLAUDE.md — Fathom Project Memory

Project memory for Claude Code sessions on the Fathom build. Read at the start of every session.

## Project framing

- **Product:** Fathom — the modern command-and-control platform for undersea warfare. Sprint 1 of Phase 0 is in progress.
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
- **Plan files** live under `~/.claude/plans/`. The Sprint 1 runbook is `users-keith-documents-claude-projects-u-rustling-kahan.md`.
- **Commit cadence:** one commit per cluster, message format `sprint1: <cluster name>`. No `--no-verify` skips. Don't amend; create new commits.
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

- Sprint plan being executed → `~/.claude/plans/users-keith-documents-claude-projects-u-rustling-kahan.md`
- DeepShip layout → flat (`<class>/<numeric_id>.wav`); the loader treats `recording_id` as `vessel_id` for vessel-level splits
- Phase 0 exit gate → operator review of LOFAR grams (Keith eyeballs `artifacts/sprint1_sanity/INDEX.md` against ICP intuition)
- Sprint 1 substrate that Sprint 2+ builds on:
  - `src/fathom/models.py` — typed contracts (Contact has multi-source provenance from Day 1)
  - `src/fathom/events.py` — Topic enum + EventBus stub (Phase 1+ → Kafka/Redpanda)
  - `src/fathom/audit.py` — provenance & sidecar writer (every artifact carries one)
  - `src/fathom/grams/normalization.py` — split-window normalization (Sprint 2 TPSW two-pass tunes on top)

## Anti-patterns to avoid

- Don't reproduce song lyrics or large copyrighted excerpts from source datasets.
- Don't propose using mel-scale, log-mel, or any perceptually-weighted frequency representation for operator-facing grams.
- Don't ship classified data or secret-tagged files. Don't `git add` files matching the tripwire patterns even if Keith requests it.
- Don't commit without an explicit Keith green light.
- Don't add framework dependencies (FastAPI, MLflow, Hydra, etc.) preemptively. Add when first needed.
