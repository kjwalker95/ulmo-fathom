# Sprint 6 Cluster F — recruitment status

Status as of 2026-05-25 (Sprint 6 close).

## Decision: recruitment deprioritized for Sprint 6

The Sprint 6 plan §F originally scoped recruitment of 2-3 additional cleared IUSS-network operators with acceptance of "at least 1 additional operator recruited (beyond CEO)." CEO call: deprioritize for Sprint 6.

Rationale:
- Sprint 5 Cluster C6 (CEO single-operator review) already established the PCD v4.3 §13.1 operator-recognition baseline qualitatively (verdict: *"normally gets what I would have got, one or two lines I did not see, few false positives"*).
- The Sprint 6 ensemble has been characterized analytically (Gate 2 verdict at commit `d2aaf55`, PARTIAL PASS).
- Operator-panel value-add lands strongest after Sprint 7's planned classification + signature-library work — at that point, panelists evaluate detection AND classification together, which is the operationally meaningful surface.

## Sprint 7+ outreach plan

Recruitment + panel execution flow into Sprint 7 per the original plan's risk language: *"Cleared IUSS operators are a small population and may not be available on sprint timescale. Mitigation: the protocol and materials are the Sprint 6 gate, not the panel results. Results land when they land."*

CEO will reach out to network when Sprint 7 classification + signature library reaches a panel-ready state.

## What Cluster F still shipped in Sprint 6

- `scripts/sprint6_panel_render.py` — ensemble-aware render with the new disagreement-overlay visual primitive (gram + member-disagreement variance heatmap). Reusable platform-layer code for any future panel.
- 15 rendered PNGs across 5 C4/C6 subset recordings (Cargo/41, Passengership/23, Passengership/32, Tanker/5, Tug/40) at the C3 winning operational threshold (bin=0.001).
- `f_self_review.md` — CEO self-review of the 15 renders (Sprint 5 C6 pattern extended to the ensemble surface).