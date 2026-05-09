# Design Memo A3: Sim-to-Real Evaluation Methodology

**Status:** Draft pending CEO sign-off (Sprint 4 Cluster A3 deliverable).
**Author:** Claude (drafted from literature digest dispatched 2026-05-09); CEO reviews and signs off below.
**Scope:** Synthetic-to-real ratio sweep methodology + operator forced-choice A/B + diagnostic distributional metrics. Operationalized in Sprint 5 (real-data ML training + ratio ablation) and Sprint 6 (calibration validation).
**Predecessor:** PCD v3 §5.1 (calibrated uncertainty), §7.4 (synthetic generator), §12.2 (vessel-level holdout); Phase1_Plan.md, Sprint4_Plan.md
**Successor:** Sprint 4 Cluster C3 (synthetic-only baseline metrics) and Sprint 5 ratio-sweep ablation execute against this memo's §7 Decisions.

---

## 1. Problem statement

Tuor's Phase 1 ML line detector trains on a synthetic + real LOFAR mix. Real data is severely limited: 63 DeepShip vessels split 43/8/12 train/val/test (val=0 Tug per Phase 0 stratification limitation), plus 4 ShipsEar smoke recordings. PCD v3 §7.4 commits to monitoring the synthetic-to-real ratio as an explicit hyperparameter and characterizing the tradeoff via ablation.

Two questions this memo answers:

1. **How do we choose a defensible synthetic-to-real training ratio** with high confidence that the chosen ratio actually optimizes real-data performance, given that we have very few real test points?
2. **How do we know the synthetic data is "realistic enough"** before scaling up the synthetic generator's output volume? — equivalently, how do we measure the sim-to-real gap and act on it?

PCD v3 §7.4 specifies a 50/50 default ratio "refined empirically." The literature digest dispatched for this memo (Liang et al. 2024 EUSIPCO; Ghosh et al. 2025 Synthio at ICLR; Bialer & Haitman 2024 RadSimReal at CVPR; Hilmes et al. 2024 SynData4GenAI; Liu et al. 2023 *Applied Acoustics*; Gui et al. 2025 SPPI; Jayasumana et al. 2024 CVPR; Tailleur et al. 2024 EUSIPCO; Heyer et al. 2023 HAL; Shi et al. 2026 in press) showed that the 50/50 default is unlikely optimal — robotics co-training literature predicts ~38% (1/φ) optimal with U-shaped degradation around it, and **representation alignment between synthetic and real distributions explains roughly 2.5× more variance than the ratio itself**. The framing question is "make synthetic look like real on the features the model uses" first; "tune the ratio" a distant second.

This memo decides the evaluation methodology that operationalizes the answers. A1 handles the synthetic generator architecture; A2 handles the ML detector architecture.

## 2. Alternatives considered

**(a) Single 50/50 train + test, no ratio sweep.** The PCD v3 §7.4 default literally interpreted. **Rejected:** picks one point on what the robotics co-training literature shows is a U-shaped curve. We can't verify we're near the optimum. Cheap but methodologically weak.

**(b) Distributional metric only (FAD / FID / MMD / Wasserstein).** Compute distance between synthetic and real distributions in some embedding space; iterate generator until distance drops below threshold. **Rejected as primary gate.** Tailleur 2024 (EUSIPCO) and Jayasumana 2024 (CVPR) converge: distance metrics depend heavily on embedding choice, the Gaussian assumptions are fragile, small reference corpora produce biased estimates. Heyer 2023 (HAL) specifically validates that Proxy-A-Distance "detects shift well but does not reliably predict the accuracy delta." Distance metrics are diagnostic, not evaluative.

**(c) Operator forced-choice A/B only (Blizzard-Challenge MOS-style).** Have CEO + 1-2 cleared peers do forced-choice "real vs synthetic" identification on N paired LOFAR clips. **Rejected as sole gate** — anchors only on perceptual realism, doesn't ground in downstream task performance. Necessary but not sufficient.

**(d) Ratio sweep + operator A/B + distributional triage (chosen).** Combine all three:
- **Primary gate:** ratio sweep ablation under vessel-level holdout, real-test detection P/R + calibration, U-curve characterized.
- **Secondary gate:** operator forced-choice A/B per Blizzard-MOS protocol; above-chance discrimination = generator iteration required.
- **Diagnostic signal (logged, not gating):** FAD with WavLM-Base+ embedding (Tailleur 2024 most-stable choice) + a domain-trained encoder; PAD with linear SVM. Tracked across generator iterations to surface synthetic-artifact regressions.

Each piece compensates for the others' weaknesses. Sweep grounds in real-data accuracy; A/B grounds in perceptual realism; distributional metrics flag distribution shift the other two might miss. **Chosen.**

## 3. Chosen approach

### 3.1 Primary gate: synthetic-to-real ratio sweep

**Sweep grid:** 0% / 25% / 38% / 50% / 75% / 100% synthetic in training mix. The 38% point is included specifically because the robotics co-training literature (Maddukuri et al. 2025) predicts ~1/φ ≈ 0.618 real (so 38.2% synthetic) as the empirical optimum across many co-training tasks. If the U-curve confirms this for LOFAR line detection, that's a publishable finding; if it disconfirms, we want to know.

**Methodology per cell:**

1. Train ML detector (per A2 memo §3 — ResNet-18 patch-CNN as primary, U-Net+clDice as parallel) on the cell's synthetic+real mix, with synthetic data drawn from the A1-memo-spec generator.
2. Real-data partition: vessel-level holdout per Sprint 3's frozen `SplitManifest` (`artifacts/sprint3_splits/deepship_splits.json`, SHA256-locked). Train uses train_vessels (43); validation uses val_vessels (8); test uses test_vessels (12). Vessel-level discipline is non-negotiable per PCD v3 §12.2.
3. Synthetic data partition: independent train/val/test splits within synthetic generator output (seeded; never shared between cells in the sweep — each cell's synthetic test set is held out from training).
4. Evaluate on **real test set only** for the headline number. Synthetic test set numbers logged but secondary.

**Metrics per cell:**

- **Real-test precision / recall / F1** at the line level (per A2 memo §5 evaluation methodology preview).
- **Real-test calibration ECE** at the per-class level (probability calibration even pre-conformal). Sprint 6 formalizes this; Sprint 5 reports raw ECE as a quality-of-output signal.
- **Real-test line-IoU** (custom: temporal-extent overlap × frequency-tolerance bin distance — defined precisely in Sprint 5 implementation).
- **Per-class breakdown:** Cargo / Passenger / Tanker / Tug. Tug class has val=0 vessels per Sprint 3 split limitation; report Tug calibration on test set with explicit "small-sample" caveat.

**Cell time and cost:**

Per-cell cost: ~2 GPU hours on g5.xlarge ($1/hr) for primary architecture training; +2 GPU hours for parallel U-Net architecture if A2 sweeps both. 6 cells × 4 GPU hours = 24 GPU hours total ≈ $24. Trivial; budget within Phase 1 plan §7's <$200 estimate.

**Acceptance criterion for Sprint 5 ratio-sweep:**
- The U-curve must show a clear minimum (i.e., a ratio that beats both endpoints). If 0%-synthetic (real-only) is the best cell, the synthetic generator failed; iterate on A1 memo's generator design.
- The chosen ratio (the U-minimum) becomes Sprint 5/6/7's frozen training-mix default. PCD v3 §7.4 amendment if not 50%.

### 3.2 Secondary gate: operator forced-choice A/B (Blizzard-MOS protocol)

**Protocol:**

1. Generate N=20-30 paired LOFAR clips: N synthetic clips (sampled across the synthetic generator's parameter space — different SNRs, different propagation environments, different biological-distractor densities) and N real clips (sampled from local DeepShip subset).
2. Render each clip as a labeled gram PNG using the production `display/render.py` pipeline. Metadata stripped — operator sees gram only, no audio.
3. Operator (CEO + 1-2 cleared peers if available) sees pairs in randomized order; chooses "real" or "synthetic" for each clip independently. No feedback during the panel.
4. Compute discrimination accuracy. Above-chance (>50% + binomial confidence interval) = fail; iterate on A1 memo's synthetic generator.

**Pass criterion:** discrimination accuracy ≤55% (within binomial 95% CI of chance for N=30 trials). Below-chance discrimination is not a failure (it just means the operator is guessing); above-chance is the failure mode.

**When to run:** at the close of Sprint 4 C1 (synthetic generator first ships); at the close of Sprint 5 if ratio-sweep shows the synthetic-only cell underperforms (suggests realism problem); optionally at Sprint 6 if calibration metrics drift unexpectedly.

**Why N=20-30:** Blizzard Challenge protocols typically use N=15-30 paired clips per evaluator; binomial 95% CI of chance on N=30 is roughly ±18%. Larger N is better but bottlenecked by operator time. CEO time budget per Phase1_Plan §7: ~1-2 hours per panel.

### 3.3 Diagnostic signal: distributional metrics (logged, not gating)

Computed on every generator iteration; logged in `artifacts/sprint5_ratio_sweep/distributional_metrics.md`. Not a pass/fail gate; flag synthetic-artifact regressions when distance grows generator-iteration-over-iteration.

**Two embeddings (Tailleur 2024 recommendation):**

1. **WavLM-Base+** (general-purpose audio embedding; Tailleur 2024 most-stable choice across speech-domain FAD studies). 768-dim embedding per clip.
2. **Domain-trained encoder** — a small encoder pretrained on DeepShip via contrastive learning (SimCLR-style) on real-only data, used as a domain-specific projection space. Produces embeddings tuned to the LOFAR distribution.

**Two metrics per embedding:**

1. **FAD (Fréchet Audio Distance):** Gaussian-fit distance between synthetic and real embedding distributions. Standard but Gaussian-assumption-fragile per Jayasumana 2024.
2. **PAD (Proxy-A-Distance):** train a linear SVM to discriminate synthetic from real in embedding space; PAD = 2(1-2ε) where ε is SVM error rate. PAD = 0 when distributions identical, PAD = 2 when perfectly separable. Heyer 2023 specifically validates PAD on acoustic-deployment-shift problems.

Total: 4 distributional metric values per generator iteration. Logged with timestamps to flag regressions.

### 3.4 Pretrain-then-finetune vs mix-and-train ablation

Two training regimes evaluated in the ratio sweep:

- **Mix-and-train (the PCD v3 §7.4 default):** synthetic + real combined into one training set; single training run with the cell's ratio.
- **Pretrain-then-finetune:** pretrain on 100% synthetic (large dataset), then finetune on 100% real (small dataset, vessel-level train split). RadSimReal (Bialer & Haitman 2024 CVPR) and the RF-localization paper (Liu et al. 2025) both report large gains from this regime over mix-and-train.

Sprint 5 ablation: test mix-and-train at all 6 ratio cells AND pretrain-finetune at the 100% / 100% endpoint (i.e., always pretrain on full synthetic, always finetune on full real). Plus one optional cell: pretrain on synthetic + finetune on a 50% synthetic + 50% real mix (sometimes wins per Synthio 2025).

Decision rule: if pretrain-then-finetune dominates the U-curve cleanly, it becomes the Sprint 6/7 production training regime. If mix-and-train wins, the U-minimum cell from §3.1 wins. If ambiguous, default to pretrain-then-finetune (better generalization properties documented across multiple domains).

### 3.5 Calibration transfer check (Sprint 6 input)

The synthetic+real mixed training raises a calibration question: does calibrated confidence learned partly on synthetic data transfer to real data? This is a Phase 1 Sprint 6 concern but A3 specifies the methodology now so Sprint 6 doesn't relitigate.

**SPPI (Synthetic-Powered Predictive Inference; Gui et al. 2025, arXiv:2505.13432):** conformal-prediction extension that incorporates synthetic data via a score transporter, preserving finite-sample coverage on the real test distribution. Direct fit for PCD v3 §5.1's calibrated-uncertainty platform commitment. Sprint 6 implements SPPI with the conformal calibration set drawn from real-only val data; synthetic data informs the score transporter but does NOT enter the calibration set itself.

This preserves PCD v3 §12.2 calibration target: per-class ECE < 0.05 on the real test distribution, conformal coverage tracks nominal alpha within finite-sample bounds — even when training data is synthetic-heavy.

## 4. Open methodology questions

These are explicitly unresolved at this memo's draft and surfaced for CEO awareness:

1. **Calibration ECE confidence intervals on val=8 vessels.** With 8 validation vessels, per-class ECE has wide bootstrap confidence intervals — Cargo (1 val vessel) is essentially uncomputable; Tug (0 val vessels) is uncomputable. **Plan:** report ECE with explicit CI bounds; flag small-class cells; defer high-confidence calibration claims to Phase 2 self-collected data.
2. **Per-class ratio sweep vs global ratio sweep.** Optimal synthetic ratio may differ by class (Cargo has 12 vessels; Tug has 3). **Plan:** Sprint 5 runs the global sweep first (one ratio across all classes). If per-class breakdown shows divergent winners, Phase 2 may run per-class sweeps. Phase 1 ships the global winner.
3. **Operator-eyeball N=20-30 statistical power.** Above-chance discrimination at α=0.05 requires N=30+ for moderate effect sizes. CEO + peers may not provide N=90 evaluator-trials per panel. **Plan:** Sprint 5 reports panel results with binomial CI; treat near-chance result (e.g., 55% with wide CI) as inconclusive rather than pass.
4. **Domain-trained encoder cold-start.** Training the §3.3 contrastive encoder on real-only data requires a Sprint 5 cluster of its own (~1-2 days work). **Plan:** Sprint 5 plan adds a small cluster for it; if too costly, fall back to WavLM-Base+ only as a single embedding.
5. **SPPI library availability.** SPPI is brand-new (arXiv 2025). Reference implementation may not exist. **Plan:** if SPPI is too rough, fall back to standard split-conformal at Sprint 6 with explicit note about residual synthetic-distribution-shift risk.

## 5. Risks and mitigations

1. **Ratio sweep U-curve doesn't show a minimum.** If the curve is monotonic (e.g., 100% real always wins), synthetic generator is failing to add useful signal. **Mitigation:** A1 memo's iteration loop kicks in — re-examine NOAA station selection, Watkins curation, KRAKEN environment diversity. The 100%-real cell becomes the floor we have to beat with better synthetic.
2. **Operator A/B above-chance discrimination.** Synthetic looks visibly different from real. **Mitigation:** specific differences logged in panel debrief; A1 generator iterates targeting those differences. Common failure modes: missing biological transients, ambient noise too smooth (lacking impulsive distractors), missing convergence-zone gain striations, simulator artifacts at frame edges.
3. **Distributional metrics diverge from downstream accuracy.** PAD says distributions match but ratio sweep shows synthetic-heavy cells fail. **Mitigation:** trust the ratio sweep (real-test accuracy is ground truth). Flag the divergence in Sprint 5 retro; consider adding embeddings or alternative metrics in Phase 2.
4. **Pretrain-then-finetune wins by a small margin.** Could be noise on small val set. **Mitigation:** report bootstrap CIs; require ≥3% F1 advantage with non-overlapping CIs to call pretrain-finetune the winner. Otherwise default to mix-and-train per PCD v3 §7.4 explicit prior.
5. **SPPI score transporter fails on small calibration set.** Synthetic-real score-transport may need more calibration data than 8-12 vessels provide. **Mitigation:** Sprint 6 ablation tests SPPI on synthetic-only data first (where calibration set can be made arbitrarily large); gauges whether the technique is methodologically sound before applying to real data.
6. **Calibration on Tug is genuinely impossible.** val=0 means no Tug calibration on val. **Mitigation:** report Tug calibration on test set with explicit small-sample caveat; do not claim calibration coverage guarantees for Tug class until Phase 2 self-collected data lands.

## 6. References

- Liang, J., Nolasco, I., Ghani, B., Phan, H., Benetos, E., Stowell, D. (2024). "Mind the Domain Gap: A Systematic Analysis on Bioacoustic Sound Event Detection." *EUSIPCO 2024*. arXiv:2403.18638.
- Ghosh, S. et al. (2025). "Synthio: Augmenting Small-Scale Audio Classification Datasets with Synthetic Data." *ICLR 2025*. arXiv:2410.02056.
- Bialer, O. & Haitman, Y. (2024). "RadSimReal: Bridging the Gap Between Synthetic and Real Data in Radar Object Detection With Simulation." *CVPR 2024*. — Pretrain-then-finetune wins on radar; informs §3.4.
- Hilmes, B. et al. (2024). "On the Effect of Purely Synthetic Training Data for Different ASR Architectures." *SynData4GenAI 2024 (Interspeech workshop)*. arXiv:2407.17997.
- Liu, Y., Yi, T. et al. (2023). "Data augmentation method for underwater acoustic target recognition based on underwater acoustic channel modeling and transfer learning." *Applied Acoustics* 211. DOI: 10.1016/j.apacoust.2023.109552.
- Gui, S., Schuckers, S. et al. (2025). "Synthetic-Powered Predictive Inference (SPPI)." arXiv:2505.13432. — Conformal extension for synthetic-augmented training; basis for §3.5.
- Jayasumana, S. et al. (2024). "Rethinking FID: Towards a Better Evaluation Metric for Image Generation." *CVPR 2024*. — FID's Gaussian-assumption fragility; argues CMMD as alternative; informs §3.3 cautious framing.
- Tailleur, M. et al. (2024). "Correlation of Fréchet Audio Distance with Human Perception." *EUSIPCO 2024*. — Embedding-stability ranking across FAD variants; basis for WavLM-Base+ recommendation in §3.3.
- Heyer, S. et al. (2023). "Investigating the usage of Proxy-A-Distance as a measure of dataset shift detection and quantification in an automotive booming-noise classification setting." HAL: hal-04166097. — PAD validation in acoustic-deployment-shift; basis for using PAD as diagnostic in §3.3.
- Shi, J. et al. (2026, in press). "Understanding Fréchet Speech Distance for Synthetic Speech Quality Evaluation." arXiv:2601.21386.
- Maddukuri, S. et al. (2025). "The Science of Co-Training: Sim-and-Real for Robot Manipulation." — Robotics co-training findings: optimal real-data weight ≈ 1/φ ≈ 0.618; informs §3.1 sweep-grid 38% inclusion.

## 7. Decisions (Sprint 5 ratio-sweep + Sprint 6 calibration spec)

1. **Primary gate (ratio sweep):** 6 cells at 0/25/38/50/75/100% synthetic; vessel-level holdout per Sprint 3 `SplitManifest`; real-test P/R/F1 + calibration ECE + line-IoU per cell; per-class breakdown.
2. **Mix-and-train vs pretrain-then-finetune ablation:** mix-and-train on all 6 cells; pretrain-on-100%-synthetic-finetune-on-100%-real as 7th cell; optionally pretrain-on-synthetic-finetune-on-50/50 as 8th cell.
3. **U-minimum cell becomes the Sprint 6/7 frozen training-mix default.** PCD v3 §7.4 amendment if ≠ 50%.
4. **Secondary gate (operator A/B):** N=20-30 paired clips, Blizzard-MOS protocol, CEO + cleared peers if available; pass = discrimination accuracy ≤ 55% with binomial 95% CI overlapping chance.
5. **Diagnostic signals:** FAD with WavLM-Base+ + domain-trained encoder; PAD with linear SVM. Logged per generator iteration; flag regressions.
6. **Calibration transfer check (Sprint 6):** SPPI (Gui 2025) for synthetic-augmented conformal calibration; conformal calibration set drawn from real-only val. If SPPI is too rough to integrate, fall back to standard split-conformal with documented residual-shift risk.
7. **Per-class ECE reporting:** explicit small-sample CI on Cargo (1 val vessel) and Tug (0 val vessels); calibration coverage claims for Tug deferred to Phase 2.
8. **Architectures swept:** both A2-memo architectures (ResNet-18 patch-CNN + U-Net+clDice). Sprint 5 reports cell results for both; whichever wins on the U-minimum cell becomes Sprint 6/7 production.
9. **Pre-training synthetic-fidelity gate:** distributional metrics computed before any ratio-sweep training; A1 generator iteration loop terminates when both primary gate (ratio sweep finds U-min that beats real-only) and secondary gate (A/B discrimination ≤ chance) pass. Distributional metrics are advisory.
10. **Outputs:** `artifacts/sprint5_ratio_sweep/INDEX.md` with per-cell results table, U-curve plot, A/B panel debrief, distributional metrics log. Phase 1 plan §6 acceptance gate items 1-3 derive from these outputs.

## 8. Sign-off

**Drafted:** 2026-05-09 (Claude).
**CEO sign-off pending.** Append below upon review:

```
[CEO sign-off note here]
Date:
Decisions locked: §7 items 1-10 [accepted | revised: ...]
Comments:
```

After sign-off, this memo becomes the binding spec for Sprint 4 Cluster C3 baseline metrics + Sprint 5 ratio-sweep ablation + Sprint 6 calibration transfer check. Subsequent revisions go to a v2 memo, not in-place edits.
