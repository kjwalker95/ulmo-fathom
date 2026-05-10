# Design Memo A2: Tuor ML Line-Detection Architecture

**Status:** Signed off (revisions per `Design_Memo_Revision_Delta.md` 2026-05-09 incorporated). Binding spec for Sprint 4 Cluster C2 implementation.
**Author:** Claude (drafted from literature digest dispatched 2026-05-09; revised per CTO + external review feedback delta dated 2026-05-09).
**Scope:** Sprint 4 Cluster C2 implementation spec (`src/fathom/detection/ml.py` and the U-Net comparison module).
**Predecessor:** PCD v3 §6.7 + §7.3, Phase1_Plan.md, Sprint4_Plan.md
**Successor:** Sprint 4 Cluster C2 (full implementation), C3 (initial training), C4 (classical-vs-ML smoke test) execute against this memo's §8 Decisions.

---

## 1. Problem statement

Phase 1 ships an ML line detector that runs in parallel with the Sprint 2/3 classical pipeline (TPSW two-pass + persistence-filtered peak detection + cluster-merge). The two detectors' agreement is measured for confidence calibration in Sprint 5 (PCD v3 §6.7 Method B; agreement-confidence calibration is research, not an assumption). Phase 1's headline thesis depends on the ML detector contributing real signal — not redundant classical-style detections at marginally different operating points.

The detection task: persistent narrow tonal lines on LOFAR spectrograms. Lines appear as thin horizontal stripes (constant frequency, persistent over seconds-to-minutes), sometimes with slow drift, often overlapping (3-8 simultaneous tonals per patch is normal — harmonics + machinery + potential second vessel). Targets in the LOFAR primary view are 3 Hz–1000 Hz (Phase 1 evaluation scoped 10-1000 Hz per A1 §3.1); per-frame frequency resolution is ~1.95 Hz at the configured FFT (n_fft=16384 at sr=32 kHz); time resolution ~128 ms.

The literature digest (Han et al. 2020 DeepLofargram; Kim & Yoon 2019/2020; Bergler et al. 2022 ANIMAL-SPOT; Schall et al. 2024 baleen benchmark; Peng et al. 2025; Shit et al. 2021 clDice; Lin et al. 2017 focal loss; Ju et al. 2022 line enhancer) surfaced strong empirical convergence on convolutional architectures with small-network backbones for sparse-positive spectrogram detection. This memo locks the architecture choice; A1 handles synthetic data; A3 handles evaluation.

### Ground-truth labeling scope

Synthetic data provides exact line-level truth (per the A1 §3.3.1 truth manifest schema). DeepShip provides real acoustic distribution and vessel-level labels, but NOT line-level annotations. Real line-detection evaluation therefore requires one of:

(a) manual line annotations by a qualified operator on a small panel,
(b) synthetic injection of known lines into real ambient clips, or
(c) a clearly marked weak-label protocol (e.g., classical-detector pseudo-labels at a conservative operating point, explicitly flagged as noisy ground truth).

Phase 1 uses (b) as the primary real-evaluation method (Tier 2 in A3 §3.1). Phase 2 adds (a) on self-collected data. **The model is NOT trained or evaluated against classical-detector output as pseudo-labels in Sprint 4.**

## 2. Alternatives considered

**(a) Patch-based binary classification + frequency-axis heatmap localization (Kim & Yoon 2019/2020 lineage, extended for multi-line).** 256×256 LOFAR-gram patches. Backbone (ResNet-18 / EfficientNet-B0) → binary head ("contains line / does not") + frequency-axis heatmap head (256-dim sigmoid, multi-line capable). Kim & Yoon 2020 reports 96.2% precision / 99.5% recall at 184 ms/inference on LOFARgrams. PCD v3 §7.3 baseline. **Chosen as primary architecture.**

**(b) Whole-spectrogram U-Net with topology-preserving loss (clDice / cbDice).** Treat lines as thin foreground masks; segmentation framing. clDice (Shit et al. 2021, CVPR) is soft-skeleton-Dice variant designed for tubular structures — exactly what tonal lines are. Peng et al. 2025 (LOFARgram U-Net++) validates the U-Net family in our exact domain. cbDice (Shi et al. 2024, MICCAI) extends with boundary + radius awareness. **Chosen as parallel comparison architecture.** Not redundant with (a) — different inductive biases, different failure modes.

**(c) Object detection (Faster R-CNN, YOLO, DETR) with line-as-bounding-box.** Lines occupying 1 frequency bin × hundreds of time frames are pathological for IoU-based losses — a 1-pixel localization error halves IoU geometry. The thin-bbox failure mode is documented; GIoU/DIoU/CIoU/EIoU all degrade similarly when one box dimension approaches a pixel. **Rejected as primary.** Faster R-CNN multi-species marine mammal detector (2024) achieved 92.3% mAP for normal-aspect-ratio bounding boxes; that result does not transfer.

**(d) Per-frequency-bin sequence model.** 1D temporal model (1D CNN, GRU, small transformer) per frequency row. Captures persistence cleanly at one frequency, but breaks on drifting tonals (frequency moves across rows mid-detection). **Rejected.**

**(e) CenterNet-style heatmap regression.** Anchor-free; output heatmap of (f_center, t_start, t_end) for each line. Avoids thin-bbox IoU pathology while keeping localization regression. Feature-Enhanced CenterNet (FE-CenterNet, RS 2022) and ST-CenterNet show competitive results on small-object detection in remote-sensing imagery. **Held as Sprint 5+ alternative if (a) and (b) both underperform.** Not in Sprint 4 scope; carry forward as a Phase 2 lever.

**Convergent literature finding:** ResNet-18 backbones still beat heavier architectures (ViT, Swin Transformer) when positives are rare and the distinguishing feature is local texture (Bergler 2022 ANIMAL-SPOT; Schall 2024 baleen benchmark; Kim & Yoon 2020). ViTs need orders of magnitude more data than IUSS-equivalent corpora typically provide; Swin's hierarchical windowing is designed for object-scale features rather than 1-bin-wide lines. ResNet-18 is the right backbone choice for both architectures (a) and (b), at least for Phase 1.

## 3. Chosen approach: dual architecture

### 3.1 Primary architecture — patch-based ResNet-18 binary + frequency-axis heatmap

Mirrors Kim & Yoon 2020 / ANIMAL-SPOT lineage, extended for multi-line patches.

- **Backbone:** ResNet-18 (PyTorch torchvision pretrained on ImageNet). Adapt input layer to single-channel spectrogram patches.
- **Patch size:** 256 × 256 spectrogram bins (PCD v3 §7.3 baseline; Kim & Yoon swept 175 model × patch-size combos and 256² wins for the LOFAR scale we use).
- **Patch stride:** 128 × 128 (50% overlap) at training time; 64 × 64 (75% overlap) at inference time. Overlap at inference reduces patch-edge effects on long lines that span multiple patches.
- **Multi-line training targets:** each training patch's heatmap target is derived from ALL lines present in that patch (per the A1 truth manifest `mask_bin_indices`). A patch containing 4 simultaneous tonals has 4 activated regions in its frequency-axis heatmap. The binary classification head is positive if ANY line is present.
- **Heads:**
  - **Binary classification head.** 2-class softmax: `{no line in patch, line in patch}`. Loss: focal loss (γ=2, per Lin et al. 2017 standard for class imbalance). Alpha-balanced if positive/negative sampling shows residual imbalance after batch sampling.
  - **Frequency-axis heatmap head.** Active only on positive patches. Outputs a 256-dim sigmoid vector (one value per frequency bin in the patch), where each activated bin indicates a line center at that frequency. Loss: BCE on the heatmap target (multiple bins can be active simultaneously). This handles multiple simultaneous lines per patch (harmonics, multi-vessel, machinery lines) without the single-regression assumption. Combined loss: `L = L_classification + λ · L_heatmap`, λ=1 default.
  - **Time-extent estimation (per detected line).** For each activated frequency bin (or cluster of adjacent activated bins), estimate `t_start` and `t_end` within the patch via a lightweight 1D temporal scan on the spectrogram row at that frequency. This is a post-hoc extraction step (not a learned head), keeping the model architecture simple while recovering per-line temporal extent.
- **Class imbalance handling:** weighted oversampling of positive patches in training batches (target 50/50 positive/negative within batch); focal loss handles residual.
- **Input-layer initialization:** default option is **conv1 channel-averaging surgery** (average the 3-channel ImageNet-pretrained conv1 weights into a single-channel kernel, preserving learned spatial filters while adapting to grayscale spectrogram input). Ablation in Sprint 5: compare against (a) 3-channel replication (naive but standard), (b) scratch initialization on synthetic. Channel-averaging is a better prior than replication for spectrogram inputs because ImageNet conv1 learns color-opponent filters that don't activate meaningfully on replicated grayscale.

### 3.2 Parallel architecture — U-Net + clDice

Mirrors Peng et al. 2025 (U-Net++ for LOFARgram) with clDice topology loss for thin-line preservation.

- **Backbone:** standard U-Net with 4 encoder/decoder levels + skip connections. Single-channel input (LOFAR magnitude/dB), single-channel output (line probability mask).
- **Input scale:** whole-gram (or large tile if memory binds; e.g., 1024 × 1024 tiles for very long recordings).
- **Loss:** combined `L = L_BCE + α · L_Dice + β · L_clDice` with α, β tuned by ablation; defaults α=1, β=0.5 per clDice paper recommendations.
- **Output thresholding:** sigmoid-then-threshold at 0.5 default; mask post-processing extracts connected components and converts to `LineOfInterest` records (frequency = mean of masked freq bins; time extent = mask span; SNR derived from underlying spectrogram values within mask).
- **Why both architectures:** primary (a) is the literature consensus and gives us the patch-classification result that's directly comparable to published numbers (Kim & Yoon 2020; ANIMAL-SPOT). Parallel (b) gives us the topology-preserving inductive bias for thin lines that is intrinsically correct for the geometry. Sprint 5 measures both on real DeepShip; whichever wins becomes the Sprint 6 + Sprint 7 production architecture. Both feed the same `Topic.LINE_DETECTED` schema downstream.

### 3.3 Loss function rationale

**Why focal loss (γ=2) for the classification head:** sparse positives (lines are rare). Focal loss is the field-standard fix (Lin et al. 2017; batch-balanced focal loss extensions, PMC 2023). γ=2 is the published default for class imbalance ratios similar to ours.

**Why frequency-axis heatmap (BCE) over single-line regression:** real LOFAR patches routinely contain 3-8 simultaneous tonals (harmonics + machinery + potential second vessel). Single-line regression assumes one canonical line per patch and fails on multi-line cases. The 256-dim sigmoid heatmap natively handles multiple lines: each activated bin is a detection. Post-hoc time-extent extraction recovers per-line temporal span without forcing the model architecture to be more complex.

**Why Dice + clDice for segmentation:** standard BCE alone produces masks that "thin out" exactly where lines need to be preserved as continuous structures. clDice's soft-skeleton ensures topological connectivity; combined with Dice for region overlap, this is the right loss for tubular foreground (Shit et al. 2021).

**Why NOT IoU-style box regression:** thin-bbox pathology (§2 alternative c). Even GIoU/DIoU/EIoU don't fix it. We avoid the framing entirely.

### 3.4 Tuor namespace decision

Sprint 4 plan §3.4 surfaced this question. The codebase reorg into `src/fathom/` (platform) + `src/tuor/` (product) was deferred indefinitely at Phase 0 exit (Phase0_Review §6 item 4). ML line detection is genuinely Tuor-product code (PCD v3 §6.7).

**Decision:** Keep ML detection at `src/fathom/detection/ml.py` for Sprint 4. Same module path as the classical detector (`src/fathom/detection/lines.py`). Deferring physical reorg matches the Phase 0 decision and avoids triggering a codebase-wide rename mid-Sprint-4.

The platform/product split remains documented in CLAUDE.md and PCD v3 §6 as a logical distinction. Physical reorg defers until either (a) the deployment topology forces it (Sprint 7 service decomposition), or (b) the second-product Day-90 gate (PCD v3 §15.2) lands and we have a concrete second product whose code organization makes the reorg unambiguous.

### 3.5 Inference-time integration with classical pipeline

ML detector runs alongside classical via `detect_lines` orchestrator extension. Both publish to `Topic.LINE_DETECTED` with respective `detection_method` enum values (`CLASSICAL`, `ML`). Sprint 5 adds the `FUSED` method when agreement-confidence calibration lands.

`LineOfInterest` schema is shared (PCD v3 §6.7 Topic.LINE_DETECTED schema-stable-across-methods commitment). No model changes required for the Pydantic contract.

## 4. Training schedule and hyperparameters (Sprint 4 C3)

Sprint 4 Cluster C3 trains primary + parallel architectures on synthetic-only data:

- **Synthetic dataset size:** 20,000 LOFAR-gram patches per epoch (Peng et al. 2025 used 20k for U-Net++ pretraining; matches the scale where small CNNs converge cleanly without overfitting to seeded noise).
- **Train/val split:** 80/20 within synthetic. Real DeepShip data not used in Sprint 4 — only Sprint 5 onward (real-data training is C5's deliverable per Phase1_Plan §5.2).
- **Batch size:** 64 patches per batch (single GPU). Adjust if GPU memory binds.
- **Optimizer:** AdamW, learning rate 1e-3 with cosine annealing over 50 epochs. Weight decay 1e-4.
- **Augmentation:** random time-flip (legitimate for line detection — direction-of-time doesn't matter for tonals), random small frequency shift (±2 bins; mimics drift at small scale), random additive Gaussian on the spectrogram patch (small σ; simulates SNR variability). NO frequency masking or time masking (would obscure the lines we're trying to detect).
- **Reproducibility:** PyTorch seed pinned; data-loader seed pinned; deterministic algorithms enabled where supported. Container parity may not hold bit-exactly for trained model weights (CUDA float drift); that's acceptable per the Sprint 2 corrected parity definition (1e-3 dB tolerance applies to detection outputs, not model weights).

Performance anchors from literature:
- DeepLofargram (Han et al. 2020): -24 dB SNR detectability on simulated LOFARgrams.
- Kim & Yoon 2020 patch-CNN: 96.2% precision / 99.5% recall on simulated LOFARgrams.
- ANIMAL-SPOT (Bergler 2022): 97.9% mean test accuracy, 95.9% AUC across 10 species (related task).
- Schall 2024 baleen benchmark: ANIMAL-SPOT (ResNet-18) wins F=0.83.

Sprint 4 acceptance criterion (per Sprint4_Plan §6 item 3): ≥80% recall of injected tonals at SNR ≥ 8 dB on synthetic test set. Conservative against the literature anchors; deliberately undershoot to avoid celebrating sprint-exit numbers that real-data evaluation will deflate.

## 5. Evaluation methodology preview

Full evaluation methodology lives in `A3_sim_to_real_evaluation.md`. Sprint 4 specifically:

- **Synthetic test set (A3 Tier 1):** held-out 20% of synthetic dataset. Report per-SNR-bucket precision/recall/F1; report line-IoU (custom metric defined in A3 §4 item 6).
- **Round-trip injection:** inject tonals at known `(freq, SNR, persistence)`, verify model recovers within frequency tolerance (2 bins) and time tolerance (8 frames).
- **Classical-vs-ML smoke (Sprint 4 C4):** both detectors run on a single Sprint 3 sample recording (Cargo/41 or Tug/9); both publish to `Topic.LINE_DETECTED`; subscribed handler logs counts. No agreement measurement yet — that's Sprint 5 C1.

**Training label source:** A1 truth manifest (§3.3.1) provides exact multi-line ground truth per synthetic patch. Real-data evaluation uses Tier 2 methodology (synthetic injection into real ambient) per A3 §3.1. The model is NOT trained or evaluated against classical-detector output as pseudo-labels in Sprint 4.

## 6. Risks and mitigations

1. **Class imbalance not fully fixed by focal loss + batch balancing.** **Mitigation:** if Sprint 4 C3 training fails to converge, add hard-negative mining (sample patches that current model classifies as positive but should be negative). Standard technique; library-supported.
2. **Multi-line patches in real data.** Real LOFAR patches routinely contain 3-8 simultaneous tonals (harmonics + machinery + potential second vessel). The frequency-axis heatmap head handles this natively. **Risk:** if adjacent-frequency lines blur into a single heatmap peak, line count is undercounted. **Mitigation:** post-hoc peak-finding on the heatmap with minimum peak separation = 2 bins (matching the A3 frequency-tolerance metric). Ablation on synthetic multi-line patches validates separation performance before real-data training.
3. **Drift handling in patch framing.** Patches are temporally short (~32 seconds at 50% overlap); slow drift over many seconds appears as a drifting line WITHIN one patch. **Mitigation:** the heatmap activates frequency bins along the drift trajectory; post-hoc time-extent extraction handles the per-frequency time scan. Cross-patch drift handled by downstream stitching.
4. **Computational cost of dual-architecture training.** ResNet-18 + U-Net dual training roughly doubles GPU time. **Mitigation:** Sprint 4 C3 trains both in parallel on a single GPU (ResNet-18 forward+backward is small enough that two models fit). If memory binds, train sequentially. Cloud GPU budget per Phase1_Plan §7 is $50-100 across Sprint 4 — comfortable margin.
5. **Sim-to-real gap on architecture choice.** A model that wins on synthetic-only may lose on real-data evaluation in Sprint 5. **Mitigation:** Sprint 4's synthetic-only training is intentionally a baseline, not a final result. Sprint 5 retrains on the synthetic+real mix and re-runs the architecture comparison; if the winner shifts, Sprint 6 onward uses the Sprint 5 winner.
6. **Conv1 channel-averaging surgery may not preserve enough learned signal.** **Mitigation:** A2 §3.1 specifies channel-averaging as default with ablation against 3-channel replication and scratch-init in Sprint 5. If channel-averaging underperforms, fall back to the better-performing alternative.
7. **U-Net + clDice training instability.** clDice can be numerically delicate (soft-skeleton iteration count is a hyperparameter). **Mitigation:** start with the published clDice reference implementation; use BCE+Dice for warmup epochs, add clDice loss term after epoch 5 once base model has converged.
8. **Heatmap post-hoc time-extent extraction may underestimate persistence on faint lines.** **Mitigation:** time-extent scan threshold tuned on synthetic data with known persistence; cross-validate against A1 truth manifest.

## 7. References

- Han, Y., Li, Y., Liu, Q., Ma, Y. (2020). "DeepLofargram: A deep learning based fluctuating dim frequency line detection and recovery." *J. Acoust. Soc. Am.* 148(4). DOI: 10.1121/10.0002172.
- Kim, J., & Yoon, B.-S. (2020). "Deep CNN Architectures for Tonal Frequency Identification in a Lofargram." *Int. J. Control, Autom. Syst.* (Springer). DOI: 10.1007/s12555-019-1014-4.
- Bergler, C. et al. (2022). "ANIMAL-SPOT enables animal-independent signal detection and classification using deep learning." *Scientific Reports* 12, 21966. DOI: 10.1038/s41598-022-26429-y.
- Schall, E. et al. (2024). "Deep learning in marine bioacoustics: a benchmark for baleen whale detection." *Remote Sensing in Ecology and Conservation*. DOI: 10.1002/rse2.392.
- Peng, D., Xu, X., Song, W., Gao, D. (2025). "Preprocessing LOFARgram through U-Net++ neural network." *Frontiers in Marine Science* 12:1528111. DOI: 10.3389/fmars.2025.1528111.
- Shit, S. et al. (2021). "clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation." *CVPR 2021*. arXiv:2003.07311.
- Shi, P. et al. (2024). "Centerline Boundary Dice Loss for Vascular Segmentation (cbDice)." *MICCAI 2024*. arXiv:2407.01517.
- Lin, T.-Y. et al. (2017). "Focal Loss for Dense Object Detection." arXiv:1708.02002.
- Ju, Y. et al. (2022). "Deep-learning-based line enhancer for passive sonar systems." *IET Radar, Sonar & Navigation*. DOI: 10.1049/rsn2.12205. — Pretrain-with-line-enhancer pattern; deferred to Sprint 5 ablation per CEO decision 2026-05-09.
- Nur Korkmaz, B. et al. (2023). "Automated detection of dolphin whistles with convolutional networks and transfer learning." *Frontiers in AI* 6:1099022.

## 8. Decisions (C2 implementation spec)

1. **Module location:** `src/fathom/detection/ml.py` for the primary patch-CNN; `src/fathom/detection/ml_unet.py` for the parallel U-Net + clDice. Both under the existing `detection/` package alongside Sprint 2/3 classical modules.
2. **Primary architecture:** ResNet-18 backbone, 256×256 patches, 50% overlap at training / 75% at inference. Binary classification head (focal loss γ=2) + **frequency-axis heatmap head (256-dim sigmoid, BCE loss, multi-line capable).** Input-layer initialization via **conv1 channel-averaging surgery** from ImageNet pretrained weights (ablate 3-channel replication and scratch-init in Sprint 5).
3. **Parallel architecture:** standard U-Net (4 levels) on whole-gram tiles; loss = BCE + Dice + clDice with α=1, β=0.5; warmup-then-add clDice schedule.
4. **Class imbalance:** weighted oversampling of positive patches (50/50 within batch) for primary; clDice handles for parallel.
5. **Tuor namespace decision:** keep `src/fathom/detection/` module path; physical reorg deferred indefinitely.
6. **Training (Sprint 4 C3):** 50 epochs, AdamW LR=1e-3 with cosine annealing, batch size 64, weight decay 1e-4. Reproducible seed; container parity at 1e-3 dB tolerance applies to detection outputs but not bit-exact model weights.
7. **Inference assembly:** per-patch heatmap peaks → per-patch line candidates (frequency from peak location, time extent from temporal scan on spectrogram row) → cross-patch stitching via connected-component / persistence-aware assembly into recording-level `LineOfInterest` records. Implementation in `ml.py` as `assemble_lines_from_patches()`.
8. **Integration:** ML detector publishes to `Topic.LINE_DETECTED` with `detection_method=ML` per Sprint 4 C4. `LineOfInterest` schema unchanged from Sprint 1.
9. **Acceptance:** Sprint 4 C3 completes with ≥80% recall of injected tonals at SNR ≥ 8 dB on synthetic test set (per Sprint4_Plan §6 item 3).
10. **Phase 1 architecture bake-off:** Sprint 5 evaluates both primary and parallel on real DeepShip data; the winner becomes the Sprint 6 + Sprint 7 production architecture. Both feed identical Pydantic contracts so the swap is local.

## 9. Sign-off

**Drafted:** 2026-05-09 (Claude). **Revised:** 2026-05-09 (per `Design_Memo_Revision_Delta.md` 2026-05-09).

```
CEO sign-off:
Date: 2026-05-09
Technical direction accepted. Revisions from Design_Memo_Revision_Delta
(2026-05-09) incorporated.
Decisions locked: §8 items 1-10 [accepted]
PCD v3 §7.4 amendment: BELLHOP-only -> KRAKEN/BELLHOP hybrid (pre-computed
IRs). Evaluation scope 10-1000 Hz for Phase 1.
```

This memo is the binding spec for Sprint 4 Cluster C2 (ML detection model implementation), C3 (initial training), and C4 (classical-vs-ML parallel smoke test). Subsequent revisions go to a v2 memo, not in-place edits.
