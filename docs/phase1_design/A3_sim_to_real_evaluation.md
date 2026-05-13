# Design Memo A3: Sim-to-Real Evaluation Methodology

**Status:** Signed off (revisions per `Design_Memo_Revision_Delta.md` 2026-05-09 incorporated). Binding spec for Sprint 4 Cluster C3 baseline metrics + Sprint 5 ratio-sweep ablation + Sprint 6 calibration transfer check.
**Author:** Claude (drafted from literature digest dispatched 2026-05-09; revised per CTO + external review feedback delta dated 2026-05-09).
**Scope:** Three-tier evaluation framework (synthetic exact-truth / real-ambient injection / operator-labeled panel) + ratio sweep methodology + diagnostic distributional metrics. Operationalized in Sprint 5 (real-data ML training + ratio ablation) and Sprint 6 (calibration validation).
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

### Ground-truth labeling scope

Synthetic data provides exact line-level truth (per the A1 §3.3.1 truth manifest schema). DeepShip provides real acoustic distribution and vessel-level labels, but NOT line-level annotations. Real line-detection evaluation therefore requires one of:

(a) manual line annotations by a qualified operator on a small panel,
(b) synthetic injection of known lines into real ambient clips, or
(c) a clearly marked weak-label protocol (e.g., classical-detector pseudo-labels at a conservative operating point, explicitly flagged as noisy ground truth).

Phase 1 uses (b) as the primary real-evaluation method (Tier 2 in §3.1). Phase 2 adds (a) on self-collected data.

## 2. Alternatives considered

**(a) Single 50/50 train + test, no ratio sweep.** The PCD v3 §7.4 default literally interpreted. **Rejected:** picks one point on what the robotics co-training literature shows is a U-shaped curve. We can't verify we're near the optimum. Cheap but methodologically weak.

**(b) Distributional metric only (FAD / FID / MMD / Wasserstein).** Compute distance between synthetic and real distributions in some embedding space; iterate generator until distance drops below threshold. **Rejected as primary gate.** Tailleur 2024 (EUSIPCO) and Jayasumana 2024 (CVPR) converge: distance metrics depend heavily on embedding choice, the Gaussian assumptions are fragile, small reference corpora produce biased estimates. Heyer 2023 (HAL) specifically validates that Proxy-A-Distance "detects shift well but does not reliably predict the accuracy delta." Distance metrics are diagnostic, not evaluative.

**(c) Operator forced-choice A/B only (Blizzard-Challenge MOS-style).** Have CEO + 1-2 cleared peers do forced-choice "real vs synthetic" identification on N paired LOFAR clips. **Rejected as sole gate** — anchors only on perceptual realism, doesn't ground in downstream task performance. Necessary but not sufficient.

**(d) Three-tier evaluation framework + ratio sweep + operator A/B + distributional triage (chosen).** Combine: three-tier evaluation (synthetic exact-truth, real-ambient injection, operator-labeled panel) as the primary structure; ratio sweep on Tier 2 as the primary mechanism; operator A/B as diagnostic; distributional metrics as triage signal.

## 3. Chosen approach

### 3.1 Primary gate: three-tier evaluation framework

Real line-detection evaluation operates across three tiers with decreasing ground-truth precision:

**Tier 1 — Synthetic exact-truth test.**
Held-out synthetic test set (20% of generator output, seeded, never seen in training). Evaluates whether the model learns the generator's distribution. Ground truth: A1 truth manifest (exact line locations, frequencies, SNRs). Metrics: per-SNR-bucket precision/recall/F1, line-IoU, frequency-error histogram. **This is the Sprint 4 C3 acceptance gate (≥80% recall at SNR≥8 dB).**

**Tier 2 — Real-ambient injection test (primary real-evaluation method).**
Inject known synthetic tonals (from A1 generator, with exact truth manifests) into held-out real DeepShip/NOAA ambient clips. Evaluate model recovery of injected lines against exact injected truth. This answers: "does the model detect known lines in real noise?" Ground truth is exact (we placed the lines). Real ambient is drawn from test-vessel recordings (vessel-level holdout preserved — ambient from test vessels is acceptable because we're evaluating line detection, not vessel classification). Metrics: same as Tier 1 but on real ambient. Per-class ambient breakdown (inject into Cargo/Passenger/Tanker/Tug ambient separately to surface class-conditional detection difficulty).

**Tier 3 — Natural-real operator-labeled panel (Sprint 6+).**
Small manually-annotated panel: 3-5 real DeepShip test recordings where the CEO marks every tonal he would call operationally, with explicit "uncertain" annotations for borderline cases. This answers: "does the model find the same lines an experienced operator finds?" Ground truth is expert judgment (noisy but operationally meaningful). Qualitative failure analysis; not a P/R gate until Phase 2 when the panel grows.

**Ratio sweep operates on Tier 2.** Each ratio cell is evaluated on Tier 2 metrics (injected-line recovery in real ambient). Tier 1 is a prerequisite sanity check. Tier 3 is a Sprint 6 qualitative validation.

### 3.1.1 Ratio sweep design

**Sweep grid:** 0% / 25% / 38% / 50% / 75% / 90% / 100% synthetic in training mix. Seven cells.

- 38% included as a theory-inspired diagnostic point from synthetic/real mixing literature (robotics co-training evidence shows non-trivial optima in this region for some tasks), not as an expected universal optimum.
- 90% included because some domains show useful optima at high synthetic ratios when real data is very scarce.

**Ratio definition:** per-batch sampling probability. In a batch of 64 patches, a "50% synthetic" cell draws ~32 synthetic and ~32 real patches per batch (stochastic, not exact). This is a sampling weight, not a dataset-count ratio.

**Repeated seeds:** minimum 3 random seeds per cell. With 8-12 test vessels, single-seed results have unacceptable variance. Report mean ± std across seeds; require non-overlapping confidence intervals to call a winner.

**Ratio selection uses validation data, NOT the final test set.** The U-minimum is identified using Tier 2 metrics on val-vessel ambient (8 vessels). The 12-vessel test set is touched exactly ONCE: after the ratio is frozen, to report final confirmation numbers. This prevents test-set contamination.

**Acceptance criterion:** The U-curve must show a ratio that beats the 0% cell (real-only) with non-overlapping CIs across 3 seeds. If no cell beats real-only, the synthetic generator needs iteration (back to A1).

**Methodology per cell:**

1. Train ML detector (per A2 memo §3 — ResNet-18 patch-CNN as primary, U-Net+clDice as parallel) on the cell's synthetic+real mix, with synthetic data drawn from the A1-memo-spec generator.
2. Real-data partition: vessel-level holdout per Sprint 3's frozen `SplitManifest` (`artifacts/sprint3_splits/deepship_splits.json`, SHA256-locked). Train uses train_vessels (43); validation uses val_vessels (8); test uses test_vessels (12). Vessel-level discipline is non-negotiable per PCD v3 §12.2.
3. Synthetic data partition: independent train/val/test splits within synthetic generator output (seeded; never shared between cells in the sweep — each cell's synthetic test set is held out from training).
4. Evaluate on **Tier 2 val-vessel injection** for ratio selection. Tier 2 test-vessel injection only after the ratio is frozen.

**Metrics per cell:**

- **Tier 2 precision / recall / F1** at the line level (custom line-IoU metric per §4 item 6).
- **Tier 2 calibration ECE** at the per-class level (probability calibration even pre-conformal). Sprint 6 formalizes this; Sprint 5 reports raw ECE as a quality-of-output signal.
- **Per-class breakdown:** Cargo / Passenger / Tanker / Tug. Tug class has val=0 vessels per Sprint 3 split limitation; report Tug calibration on test set with explicit "small-sample" caveat.

**Cell time and cost:**

Per-cell cost: ~2 GPU hours on g5.xlarge ($1/hr) for primary architecture training × 3 seeds; +6 GPU hours for parallel U-Net architecture × 3 seeds if A2 sweeps both. 7 cells × 12 GPU hours total ≈ 84 hours ≈ $84. Trivial; budget within Phase 1 plan §7's <$200 estimate. (Triple cost vs original estimate due to 3-seed repetition.)

### 3.2 Secondary gate (diagnostic, not gating): operator forced-choice A/B

**Protocol:** paired forced-choice. Each trial shows one real and one synthetic LOFAR gram side-by-side (randomized left/right). Operator identifies which is real. N=20-30 trials per evaluator.

**Decision rule:**

- **FAIL** (iterate generator): lower bound of binomial 95% CI is above 50%, OR discrimination ≥70% with consistent qualitative artifact notes.
- **INCONCLUSIVE:** discrimination 55–70% with CI overlapping chance. Log operator's qualitative notes on what looks different; iterate if notes identify systematic artifacts.
- **PASS:** discrimination ≤55% with CI overlapping chance.

**Relationship to ratio sweep:** The A/B panel is **DIAGNOSTIC, not gating.** If the ratio sweep finds a clear U-minimum that beats real-only, the synthetic is adding useful training signal regardless of whether operators can spot visual differences. The A/B panel identifies what to improve in the generator, not whether to use the generator. (Rationale: synthetic data can be useful for training even if it's not perceptually identical to real data — the model may extract features humans don't consciously perceive.)

**When to run:** at the close of Sprint 4 C1 (synthetic generator first ships); at the close of Sprint 5 if ratio-sweep shows the synthetic-only cell underperforms (suggests realism problem); optionally at Sprint 6 if calibration metrics drift unexpectedly.

**Why N=20-30:** Blizzard Challenge protocols typically use N=15-30 paired clips per evaluator; binomial 95% CI of chance on N=30 is roughly ±18%. Larger N is better but bottlenecked by operator time. CEO time budget per Phase1_Plan §7: ~1-2 hours per panel.

### 3.3 Diagnostic signal: distributional metrics (logged, not gating)

Computed on every generator iteration; logged in `artifacts/sprint5_ratio_sweep/distributional_metrics.md`. Not a pass/fail gate; flag synthetic-artifact regressions when distance grows generator-iteration-over-iteration.

**Two embeddings:**

1. **PANNs (Pretrained Audio Neural Networks):** environmental/underwater audio embedding. Tailleur 2024 (EUSIPCO) found PANNs-WGM-LogMel had strongest correlation with perceptual ratings in environmental-sound FAD evaluation. More relevant to LOFAR spectrograms than speech-trained models.
2. **Domain-trained encoder:** small encoder pretrained on DeepShip via contrastive learning (SimCLR-style) on real-only data. Domain-specific projection space tuned to LOFAR distribution.

Optional tertiary: WavLM-Base+ (general audio; included for comparability with published FAD benchmarks but not primary for underwater domain).

Sprint 5 implementation uses whichever of PANNs or WavLM has cleaner PyTorch integration as the off-the-shelf choice; domain-trained encoder is the higher-value investment.

**Two metrics per embedding:**

1. **FAD (Fréchet Audio Distance):** Gaussian-fit distance between synthetic and real embedding distributions. Standard but Gaussian-assumption-fragile per Jayasumana 2024.
2. **PAD (Proxy-A-Distance):** train a linear SVM to discriminate synthetic from real in embedding space; PAD = 2(1-2ε) where ε is SVM error rate. PAD = 0 when distributions identical, PAD = 2 when perfectly separable. Heyer 2023 specifically validates PAD on acoustic-deployment-shift problems.

Total: 4 distributional metric values per generator iteration (or 6 if WavLM is logged tertiary). Logged with timestamps to flag regressions.

### 3.4 Pretrain-then-finetune vs mix-and-train ablation

Two training regimes evaluated in the ratio sweep:

- **Mix-and-train (the PCD v3 §7.4 default):** synthetic + real combined into one training set; single training run with the cell's ratio.
- **Pretrain-then-finetune:** pretrain on 100% synthetic (large dataset), then finetune on 100% real (small dataset, vessel-level train split). RadSimReal (Bialer & Haitman 2024 CVPR) and the RF-localization paper (Liu et al. 2025) both report large gains from this regime over mix-and-train.

Sprint 5 ablation: test mix-and-train at all 7 ratio cells AND pretrain-finetune at the 100% / 100% endpoint (i.e., always pretrain on full synthetic, always finetune on full real). Plus one optional cell: pretrain on synthetic + finetune on a 50% synthetic + 50% real mix (sometimes wins per Synthio 2025).

Decision rule: if pretrain-then-finetune dominates the U-curve cleanly, it becomes the Sprint 6/7 production training regime. If mix-and-train wins, the U-minimum cell from §3.1.1 wins. If ambiguous, default to pretrain-then-finetune (better generalization properties documented across multiple domains).

### 3.5 Calibration transfer check (Sprint 6 — research spike, not binding)

The synthetic+real mixed training raises a calibration question: does calibrated confidence learned partly on synthetic data transfer to real data?

**Default approach (Sprint 6 binding):** standard split-conformal prediction with the conformal calibration set drawn exclusively from real-only val data. Synthetic data enters training but NOT calibration. This preserves finite-sample coverage guarantees on the real distribution without novel methodology risk.

**Research spike (Sprint 6 optional, time-permitting):** SPPI (Synthetic-Powered Predictive Inference; Gui et al. 2025) — conformal extension that incorporates synthetic data via a score transporter. If the reference implementation exists and integrates cleanly, compare SPPI calibration efficiency against standard split-conformal. If SPPI produces tighter prediction intervals at the same coverage, adopt for Sprint 7. If integration is fragile or gains are marginal, stay with standard split-conformal.

PCD v3 §12.2 calibration target (per-class ECE < 0.05) applies to the real test distribution regardless of which conformal method is used.

## 4. Open methodology questions

These are explicitly unresolved at this memo's draft and surfaced for CEO awareness:

1. **Calibration ECE confidence intervals on val=8 vessels.** With 8 validation vessels, per-class ECE has wide bootstrap confidence intervals — Cargo (1 val vessel) is essentially uncomputable; Tug (0 val vessels) is uncomputable. **Plan:** report ECE with explicit CI bounds; flag small-class cells; defer high-confidence calibration claims to Phase 2 self-collected data.
2. **Per-class ratio sweep vs global ratio sweep.** Optimal synthetic ratio may differ by class (Cargo has 12 vessels; Tug has 3). **Plan:** Sprint 5 runs the global sweep first (one ratio across all classes). If per-class breakdown shows divergent winners, Phase 2 may run per-class sweeps. Phase 1 ships the global winner.
3. **Operator-eyeball N=20-30 statistical power.** Above-chance discrimination at α=0.05 requires N=30+ for moderate effect sizes. CEO + peers may not provide N=90 evaluator-trials per panel. **Plan:** Sprint 5 reports panel results with binomial CI; treat near-chance result (e.g., 55% with wide CI) as inconclusive rather than pass.
4. **Domain-trained encoder cold-start.** Training the §3.3 contrastive encoder on real-only data requires a Sprint 5 cluster of its own (~1-2 days work). **Plan:** Sprint 5 plan adds a small cluster for it; if too costly, fall back to PANNs only as a single embedding.
5. **SPPI library availability.** SPPI is brand-new (arXiv 2025). Reference implementation may not exist. **Plan:** standard split-conformal is the binding default per §3.5; SPPI is a research spike layered on top if integration is mature.
6. **Line-IoU metric definition.** Must be locked before Sprint 5 implementation. Proposed definition (from A2 §5 preview): for a predicted line and a ground-truth line, `line-IoU = temporal_overlap_ratio × freq_proximity_weight`, where `temporal_overlap_ratio = intersection of [t_start, t_end] intervals / union of [t_start, t_end] intervals`, and `freq_proximity_weight = 1.0 if |f_pred - f_true| ≤ 2 bins, 0.5 if ≤ 4 bins, 0.0 otherwise`. Hungarian matching for multi-line assignment within evaluation windows. **Lock this definition in Sprint 5 C1 implementation; do not defer further.**

## 5. Risks and mitigations

1. **Ratio sweep U-curve doesn't show a minimum.** If the curve is monotonic (e.g., 100% real always wins), synthetic generator is failing to add useful signal. **Mitigation:** A1 memo's iteration loop kicks in — re-examine NOAA station selection, Watkins curation, KRAKEN environment diversity. The 100%-real cell becomes the floor we have to beat with better synthetic.
2. **Operator A/B above-chance discrimination.** Synthetic looks visibly different from real. **Mitigation per §3.2:** A/B is diagnostic, not gating — it identifies generator-improvement targets. Specific differences logged in panel debrief; A1 generator iterates targeting those differences. Common failure modes: missing biological transients, ambient noise too smooth (lacking impulsive distractors), missing convergence-zone gain striations, simulator artifacts at frame edges.
3. **Distributional metrics diverge from downstream accuracy.** PAD says distributions match but ratio sweep shows synthetic-heavy cells fail. **Mitigation:** trust the ratio sweep (real-test accuracy is ground truth). Flag the divergence in Sprint 5 retro; consider adding embeddings or alternative metrics in Phase 2.
4. **Pretrain-then-finetune wins by a small margin.** Could be noise on small val set. **Mitigation:** report bootstrap CIs; require ≥3% F1 advantage with non-overlapping CIs to call pretrain-finetune the winner. Otherwise default to mix-and-train per PCD v3 §7.4 explicit prior.
5. **SPPI score transporter fails on small calibration set.** Synthetic-real score-transport may need more calibration data than 8-12 vessels provide. **Mitigation per §3.5:** standard split-conformal is the binding default; SPPI is optional research spike. No project-critical-path dependency on SPPI.
6. **Calibration on Tug is genuinely impossible.** val=0 means no Tug calibration on val. **Mitigation:** report Tug calibration on test set with explicit small-sample caveat; do not claim calibration coverage guarantees for Tug class until Phase 2 self-collected data lands.
7. **3-seed cost overrun.** 7 cells × 3 seeds × ~2 GPU hours each ≈ 42 GPU hours (~$42); add U-Net parallel ≈ $84. Within budget. **Mitigation:** monitor cumulative cost; if overruns, reduce to single-seed for non-extreme cells (25%, 50%, 75%) and triple-seed only the corners and 38% theory-inspired point.

## 6. References

- Liang, J., Nolasco, I., Ghani, B., Phan, H., Benetos, E., Stowell, D. (2024). "Mind the Domain Gap: A Systematic Analysis on Bioacoustic Sound Event Detection." *EUSIPCO 2024*. arXiv:2403.18638.
- Ghosh, S. et al. (2025). "Synthio: Augmenting Small-Scale Audio Classification Datasets with Synthetic Data." *ICLR 2025*. arXiv:2410.02056.
- Bialer, O. & Haitman, Y. (2024). "RadSimReal: Bridging the Gap Between Synthetic and Real Data in Radar Object Detection With Simulation." *CVPR 2024*. — Pretrain-then-finetune wins on radar; informs §3.4.
- Hilmes, B. et al. (2024). "On the Effect of Purely Synthetic Training Data for Different ASR Architectures." *SynData4GenAI 2024 (Interspeech workshop)*. arXiv:2407.17997.
- Liu, Y., Yi, T. et al. (2023). "Data augmentation method for underwater acoustic target recognition based on underwater acoustic channel modeling and transfer learning." *Applied Acoustics* 211. DOI: 10.1016/j.apacoust.2023.109552.
- Gui, S., Schuckers, S. et al. (2025). "Synthetic-Powered Predictive Inference (SPPI)." arXiv:2505.13432. — Conformal extension for synthetic-augmented training; basis for §3.5 optional spike.
- Jayasumana, S. et al. (2024). "Rethinking FID: Towards a Better Evaluation Metric for Image Generation." *CVPR 2024*. — FID's Gaussian-assumption fragility; argues CMMD as alternative; informs §3.3 cautious framing.
- Tailleur, M. et al. (2024). "Correlation of Fréchet Audio Distance with Human Perception." *EUSIPCO 2024*. — PANNs-WGM-LogMel as strongest embedding for environmental-sound FAD; basis for §3.3 PANNs primary recommendation.
- Heyer, S. et al. (2023). "Investigating the usage of Proxy-A-Distance as a measure of dataset shift detection and quantification in an automotive booming-noise classification setting." HAL: hal-04166097. — PAD validation in acoustic-deployment-shift; basis for using PAD as diagnostic in §3.3.
- Shi, J. et al. (2026, in press). "Understanding Fréchet Speech Distance for Synthetic Speech Quality Evaluation." arXiv:2601.21386.
- Maddukuri, S. et al. (2025). "The Science of Co-Training: Sim-and-Real for Robot Manipulation." — Robotics co-training findings: optimal real-data weight ≈ 1/φ ≈ 0.618; informs §3.1.1 sweep-grid 38% inclusion.

## 7. Decisions (Sprint 5 ratio-sweep + Sprint 6 calibration spec)

1. **Three-tier evaluation:** Tier 1 (synthetic exact-truth), Tier 2 (real-ambient injection, **primary real-evaluation method**), Tier 3 (operator-labeled natural-real panel, Sprint 6+). Ratio sweep operates on Tier 2. Sprint 4 C3 acceptance uses Tier 1 only.
2. **Mix-and-train vs pretrain-then-finetune ablation:** mix-and-train on all 7 cells (0/25/38/50/75/90/100%); pretrain-on-100%-synthetic-finetune-on-100%-real as 8th cell; optionally pretrain-on-synthetic-finetune-on-50/50 as 9th cell. Minimum 3 seeds per cell.
3. **U-minimum cell becomes the Sprint 6/7 frozen training-mix default.** PCD v3 §7.4 amendment if ≠ 50%.
3a. **Ratio selection uses val, not test.** U-minimum identified on val-vessel Tier 2 metrics. Test set touched once for final confirmation after ratio is frozen.
4. **Operator A/B (diagnostic, not gating):** N=20-30 paired forced-choice trials; fail = binomial 95% CI lower bound above 50% OR discrimination ≥70% with consistent artifact notes; pass = ≤55% with CI overlapping chance; inconclusive otherwise. A/B identifies generator improvement targets, does not gate synthetic usage if ratio sweep shows benefit.
5. **Diagnostic signals:** FAD with PANNs + domain-trained encoder; PAD with linear SVM. Optional: WavLM-Base+ for comparability. Logged per generator iteration; flag regressions. Non-gating.
6. **Calibration (Sprint 6):** standard split-conformal with real-only val calibration set (binding default). SPPI as optional research spike if reference implementation is mature. Per-class ECE < 0.05 target on real test distribution.
7. **Per-class ECE reporting:** explicit small-sample CI on Cargo (1 val vessel) and Tug (0 val vessels); calibration coverage claims for Tug deferred to Phase 2.
8. **Architectures swept:** both A2-memo architectures (ResNet-18 patch-CNN + U-Net+clDice). Sprint 5 reports cell results for both; whichever wins on the U-minimum cell becomes Sprint 6/7 production.
9. **Pre-training synthetic-fidelity gate:** distributional metrics computed before any ratio-sweep training; A1 generator iteration loop terminates when both Tier 1 sanity passes and Tier 2 ratio sweep finds a U-min that beats real-only. Distributional metrics are advisory.
10. **Outputs:** `artifacts/sprint5_ratio_sweep/INDEX.md` with per-cell results table, U-curve plot, A/B panel debrief, distributional metrics log. Phase 1 plan §6 acceptance gate items 1-3 derive from these outputs.
11. **Line-IoU metric (locked):** `line-IoU = temporal_overlap_ratio × freq_proximity_weight` per §4 item 6. Hungarian matching for multi-line assignment.

## 8. Sign-off

**Drafted:** 2026-05-09 (Claude). **Revised:** 2026-05-09 (per `Design_Memo_Revision_Delta.md` 2026-05-09).

```
CEO sign-off:
Date: 2026-05-09
Technical direction accepted. Revisions from Design_Memo_Revision_Delta
(2026-05-09) incorporated.
Decisions locked: §7 items 1-11 [accepted]
PCD v3 §7.4 amendment: BELLHOP-only -> KRAKEN/BELLHOP hybrid (pre-computed
IRs). Evaluation scope 10-1000 Hz for Phase 1.
```

This memo is the binding spec for Sprint 4 Cluster C3 baseline metrics + Sprint 5 ratio-sweep ablation + Sprint 6 calibration transfer check. Subsequent revisions go to a v2 memo, not in-place edits.


---

## Addendum: 2026-05-13 — Sprint 4 close-out deltas

**Tier-1 alone is insufficient — C4 demonstrated it (2026-05-13).** ResNet-18 patch-CNN passed nothing on Tier-1 (F1=0.151), but on real DeepShip data at training-default thresholds it produced **zero** predictions. The class-head domain gap manifests only on real-data inference, not on synthetic val. Tier-2 (real-ambient injection) and Tier-3 (operator-labeled panel) are non-optional for Phase 1 evaluation — they would have caught the class-head failure before C4. Sprint 5 must wire Tier-2 as the primary evaluation gate alongside Tier-1.

**SWellEx-96 sim-to-real CIR validation loop (Sprint 5+ candidate).** Per the 2026-05-12 canonical-IR hunt: SWellEx-96 (UCSD MPL) raw recordings cover 49-400 Hz at the SWellEx site with a published environmental model. Nannuru et al. (IEEE JOE 2022, doi:10.1109/JOE.2022.3205614) extracted measured channel impulse responses from these recordings via sparse Bayesian learning. This enables a closed-loop sim-to-real validation: generate synthetic IRs from the SWellEx environmental model, compare against CIRs extracted from raw recordings, and measure the sim-to-real gap at 49-400 Hz directly. Single public dataset where this is possible; recommended as a Sprint 5+ cluster contingent on team bandwidth.

**Sprint 5 ratio sweep collapse:** per the A2 addendum, the original "evaluate both architectures through full ratio sweep" collapses to "U-Net through full sweep + single ResNet sanity cell at winning ratio." This is a compute and sprint-time savings that the C3.h baseline result enables.