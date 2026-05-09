# Design Memo A1: Synthetic LOFAR Data Generator

**Status:** Draft pending CEO sign-off (Sprint 4 Cluster A1 deliverable).
**Author:** Claude (drafted from literature digest dispatched 2026-05-09); CEO reviews and signs off below.
**Scope:** Sprint 4 Cluster C1 implementation spec (`src/fathom/synthetic/`).
**Predecessor:** PCD v3 §7.4, Phase1_Plan.md, Sprint4_Plan.md
**Successor:** Sprint 4 Cluster B1 (spike) and C1 (full implementation) execute against this memo's §7 Decisions.

---

## 1. Problem statement

Tuor's Phase 1 ML line-detection model trains on a synthetic + real LOFAR data mix. Real data is severely limited (63 DeepShip vessels + 4 ShipsEar smoke recordings on disk; full-release acquisition stalled per Phase0_Review §6 item 1). Without realistic synthetic data, the ML detector either overfits to the small real distribution or underperforms classical at the same operating point — defeating the Phase 1 thesis that calibrated ML is the structural moat over classical-at-fixed-thresholds.

Per PCD v3 §7.4, the synthetic generator is platform-layer infrastructure (Fathom, not Tuor — future products consume the same generator). Two named risks bind the design:

- **Sim-to-real gap.** Clean synthetic noise produces models that fail on real ocean ambient. Mitigation per PCD v3 §7.4: "synthetic noise is modeled as realistic ocean noise; white noise is not used."
- **Synthetic-to-real ratio overfitting.** Too much synthetic in training causes overfitting to synthetic distribution. Mitigation per PCD v3 §7.4: "training mix is monitored as an explicit hyperparameter; ablations characterize the tradeoff."

The literature digest dispatched for this memo (Peng et al. 2025; Maddukuri et al. 2025; Liu et al. 2023; Synthio 2024; RadSimReal 2024; Haver et al. 2018; Licciardi & Carbone 2024; Bossér et al. 2024) surfaced both convergent practice and consequential debates. This memo decides the architecture; A3's memo handles evaluation methodology.

## 2. Alternatives considered

**(a) White Gaussian noise + sinusoidal tonals.** Naive baseline. Rejected per PCD v3 §7.4 explicit prohibition on white noise. Real ocean ambient is colored, with biological transients and propagation artifacts; a model trained on white-noise synthetic catastrophically fails on real recordings.

**(b) NOAA ambient + injected tonals (no biologicals, no propagation).** Better than (a) — real colored ambient. Misses biological false-alarm sources operators see in real recordings (whale calls, snapping shrimp); detection model has no exposure to confusable distractors. Rejected as insufficient.

**(c) NOAA + Watkins biologicals + simplified BELLHOP propagation.** PCD v3 §7.4's nominal architecture. Strong baseline. Limitation surfaced in literature digest: BELLHOP is ray-theory propagation, designed for high-frequency cavitation analysis. For the 3–500 Hz band where blade rate (5–12 Hz) and machinery tonals (~50 Hz) live, **normal-mode propagation (KRAKEN) is materially more accurate**, with multipath and shadow-zone effects that ray theory under-resolves at low frequency (Peng et al. 2025; LOFARgram U-Net++ paper validates this in our exact domain).

**(d) NOAA + Watkins + KRAKEN low-freq + BELLHOP high-freq + decaying-cosine ship-source primitives.** Hybrid propagation matched to frequency band. Decaying-cosine pulses with parameterized `(γ, ω₀, T, Rayleigh-jitter, harmonic structure)` per Peng et al. 2025 are the most realistic published ship-source primitive — substantially better than pure sinusoids. **Chosen.**

**(e) GAN/diffusion synthesis (Synth-SONAR, Syn2Real, AS-DCGAN).** State-of-the-art for sonar imagery; gains documented at ~60% AP improvement on side-scan sonar (Agrawal et al. 2024). Rejected for Phase 1: literature shows gains are modest and brittle for spectrogram-class data; AS-DCGAN reports 81% on DeepShip vs. 98% ensemble baselines, and progressive-frequency learning instabilities are documented. Reserve as a Phase 2 lever if KRAKEN+sample-based pipeline runs out of degrees of freedom.

## 3. Chosen approach: layered noise model

### 3.1 Layer 1 — NOAA ambient (base)

Real ocean noise sampled from the NOAA NCEI Ocean Noise Reference Station Network (Haver et al. 2018, *Marine Policy*; 12 stations; calibrated 10–2000 Hz continuous, deployed 2014+). Curated subset:

- **Geographic diversity:** at least one station each from Pacific, Atlantic, Arctic, tropical (4 of 12 minimum).
- **Recording duration:** ≥10 hours per station so we can sample without re-using clips across train/val.
- **Time-of-year diversity:** if archives permit, sample across at least 2 seasonal regimes per station (biological activity varies seasonally and is part of what the synthetic generator must model).
- **Sample-rate normalization:** resample to 32 kHz at load via the Sprint 3 `_resample.py` polyphase pipeline.

Manifest sidecar lists the specific station IDs, recording timestamps, and SHA256 hashes of each curated source clip. Audit-trail on every synthetic gram identifies which NOAA source was the base layer.

### 3.2 Layer 2 — Watkins biological transients (distractors)

Watkins Marine Mammal Sound Database (https://cis.whoi.edu/science/B/whalesounds/) overlaid as distractors at controlled rate. Curation per the WhaleNet recipe (Licciardi & Carbone 2024, *IEEE Access*):

- **Drop classes with <50 samples** (51 → ~32 species).
- **De-duplicate** (catalog has known duplicates).
- **Resample to median rate (47.6 kHz)**, not the floor — preserves high-frequency content.
- **Stratified split** before training so biological classes are seen in both train and val.

Overlay rate: synthetic gram contains 0–3 biological transients per 60-second window at random time/frequency, amplitude drawn from log-normal distribution centered on operationally-realistic SNR. Specific biological-class selection (whale calls, snapping shrimp, fish vocalizations) is a second-pass tuning decision after C1 ships first artifacts; CEO operator review ensures the chosen biologicals match real-world false-alarm sources.

### 3.3 Layer 3 — Tonal injection (positive class)

Decaying-cosine pulses per Peng et al. 2025 (*Frontiers in Marine Science*):

```
s(t) = Σ_h (a_h · exp(-γ · t) · cos(2π · h · ω₀ · t + φ_h))
       at clustered times sampled with Rayleigh-distributed within-cluster jitter
       and Gaussian inter-cluster spacing T.
```

Parameterization (per-injection schema, sampled from priors):

| Parameter | Distribution | Rationale |
|---|---|---|
| Fundamental ω₀ (Hz) | Uniform 5–500 Hz primary; uniform 500–1000 Hz secondary; biased toward 5–50 Hz | Blade-rate + machinery tonals dominate the operationally-relevant range |
| n_harmonics | Discrete uniform {1, 2, 3} (default 3 per literature) | Real ship signatures show 2-3 visible harmonics most commonly |
| Harmonic amplitude decay | Exponential, 0.3–0.7 per harmonic | Higher harmonics quieter than fundamental |
| Decay γ (1/s) | Log-uniform 0.01–1.0 | Some tonals are nearly steady; some decay over seconds |
| Cluster period T (s) | Log-uniform 1–60 | Captures slow-recurring vs. continuous tonals |
| Within-cluster jitter | Rayleigh, σ = 0.1 · T | Per Peng et al. |
| Inter-cluster jitter | Gaussian, σ = 0.05 · T | Per Peng et al. |
| Total persistence (s) | Log-uniform 1–120 | Spans below-detection (<min_persistence_s) up to long-persistence |
| Drift rate (Hz/s) | Gaussian, μ=0, σ=0.05 | Most tonals stable; some submarines drift with speed changes. Drift was rarely modeled in published synthetic — gap we are filling. |
| Per-source SNR (dB) | Log-normal, μ=8, σ=4 | Spans below threshold (<6 dB) up to strong (>20 dB) |

SNR is computed against the **local ambient at the tonal frequency**, not global RMS — same convention the Tuor classical detector uses (PCD v3 §6.3 / Sprint 2 substrate).

### 3.4 Layer 4 — Propagation artifacts (KRAKEN low-freq + BELLHOP high-freq)

Hybrid propagation modeler:

- **3–500 Hz: KRAKEN normal-mode propagation.** Pre-computed impulse responses for a small set of canonical environments (deep ocean isovelocity profile, downward-refracting summer profile, upward-refracting winter profile, shallow-water sediment-dominated) — 5–10 environments total covering the ranges PCD v3 §6.7 Method B cares about. Per-injection environment sampled uniformly. Source-receiver geometry: ranges 1–50 km, source depths 5–500 m (covers surface vessels through patrol-depth submarines).
- **>500 Hz: simplified BELLHOP ray-theory** for cavitation/broadband content. Same canonical environments; ray paths cached.

Propagation is applied to the injected tonal *before* mixing with ambient. This gives realistic multipath striations, shadow-zone effects, and convergence-zone gain variation that operators read as cues.

This is the **PCD v3 §7.4 amendment Keith approved out-of-band** — original PCD v3 §7.4 said "Bellhop-derived"; revised to "KRAKEN low-freq + BELLHOP high-freq hybrid."

### 3.5 Mixing and rendering

Final waveform = ambient + biologicals + propagation-applied tonal.

Output options:
- **Waveform output** for full-pipeline-realism training (resampled to 32 kHz).
- **Direct LOFAR-spectrogram output** (skip waveform reconstruction) for training-throughput optimization. C1 implementation defaults to waveform; spectrogram-direct is an optional optimization if training throughput becomes binding.

Reproducibility: all randomness via seeded `numpy.random.default_rng(seed)`. Each synthetic gram carries an audit sidecar with seed, NOAA source ID + clip timestamp, Watkins selection IDs, KRAKEN/BELLHOP environment ID, and full tonal-injection parameter snapshot. Manifest hash over the synthetic dataset locks reproducibility.

## 4. Evaluation methodology (preview)

Full evaluation methodology lives in `sim_to_real_evaluation.md` (A3 memo). Key gates summarized here:

- **Primary gate: ratio sweep.** 0/25/38/50/75/100% synthetic in training mix; vessel-level holdout on real DeepShip; report real-test PR + calibration error per cell.
- **Secondary gate: operator forced-choice A/B.** Blizzard-Challenge-style MOS protocol; N=20-30 paired LOFAR clips (synthetic vs real DeepShip). Above-chance discrimination = generator iteration required.
- **Diagnostic signals (not gates):** FAD with WavLM-Base+ embedding (Tailleur 2024 most-stable choice) + PAD with linear SVM on a domain-trained encoder.

The synthetic generator iteration loop terminates when both primary gate and secondary gate pass; diagnostic signals logged but not blocking.

## 5. Risks and mitigations

1. **KRAKEN parameter sensitivity.** Normal-mode propagation depends on water-depth profile, sound-speed profile, bottom type. Picking a fixed canonical environment vs. varying per-sample is a cost-vs-realism tradeoff. **Mitigation:** start with 5 environments covering the operational range; if A3 evaluation surfaces overfitting to a single environment regime, expand to 10-15. C1 implementation pre-computes impulse responses per environment so per-injection cost is modest.
2. **Watkins biological-class match.** Choosing biologicals that match real-world false-alarm sources is a domain-knowledge decision. **Mitigation:** CEO reviews specific Watkins selections at C1 close against IUSS operator memory of what biological false alarms actually look like; iterate if mismatch surfaces.
3. **NOAA archive curation cost.** Full archive is 300+ TB; we need a curated subset. **Mitigation:** start with 4-5 stations covering different ambient regimes (Pacific/Atlantic/Arctic/tropical); ~10 hours per station ≈ 50 hours total ≈ 200 GB. Expand if A3 evaluation shows overfitting to a single regime.
4. **KRAKEN library availability and licensing.** Open-source KRAKEN implementations exist (`pyat`, `arlpy`) but are research-grade. **Mitigation:** if `pyat`/`arlpy` integration is fragile, fall back to pre-computed impulse responses for the canonical environments published with established acoustic textbooks; no per-sample KRAKEN runtime.
5. **Drift-rate calibration.** Few published synthetic generators model drift; our Gaussian σ=0.05 Hz/s prior is an educated guess. **Mitigation:** A3 ratio-sweep ablation will surface whether drift modeling is helping or hurting; tune the prior in Sprint 5 if needed.
6. **Defer-GAN regret.** If KRAKEN+sample-based synthetic plateaus at sim-to-real gap larger than acceptable, GAN/diffusion is the Phase 2 lever. **Mitigation:** A3 evaluation's primary gate will surface the plateau cleanly; Phase 2 plan inherits a clear "GAN augmentation" cluster scoping if needed.

## 6. References

The literature digest sourcing this memo cited:

- Peng, D., Xu, X., Song, W., Gao, D. (2025). "Preprocessing LOFARgram through U-Net++ neural network." *Frontiers in Marine Science* 12:1528111. DOI: 10.3389/fmars.2025.1528111. — KRAKEN normal-mode + decaying-cosine source primitives; primary literature anchor for §3.3 and §3.4.
- Haver, S. M. et al. (2018). "Monitoring long-term soundscape trends in U.S. waters: The NOAA/NPS Ocean Noise Reference Station Network." *Marine Policy*. DOI: 10.1016/j.marpol.2017.11.024. — NOAA NRS canonical reference for §3.1.
- Licciardi, A., & Carbone, D. (2024). "WhaleNet: A Novel Deep Learning Architecture for Marine Mammals Vocalizations on Watkins Marine Mammal Sound Database." *IEEE Access* 12. arXiv:2402.17775. — Watkins curation recipe for §3.2.
- Maddukuri, S. et al. (2025). "The Science of Co-Training: Sim-and-Real for Robot Manipulation." — Synthetic-to-real ratio findings (~38% optimal, U-shaped degradation, structured alignment dominant); informed §4 ratio sweep.
- Liu, Y., Yi, T. et al. (2023). "Data augmentation method for underwater acoustic target recognition based on underwater acoustic channel modeling and transfer learning." *Applied Acoustics* 211. DOI: 10.1016/j.apacoust.2023.109552. — Channel-physics-based synthetic augmentation in our exact domain.
- Bossér, D., Nordenvaad, M.L., Hendeby, G., & Skog, I. (2024). "Broadband Passive Sonar Track-Before-Detect Using Raw Acoustic Data." arXiv:2412.15727. — Vector-autoregressive ambient-noise modeling validates non-Gaussian ambient.
- Synthio: Ghosh, S. et al. (2025). "Augmenting Small-Scale Audio Classification Datasets with Synthetic Data." ICLR 2025. arXiv:2410.02056. — Diminishing-returns of synthetic-augmentation as real data grows; informs expectations for our 63-vessel regime.
- Agrawal, A. et al. (2024). "Syn2Real Domain Generalization for Underwater Mine-like Object Detection Using Side-Scan Sonar." arXiv:2410.12953. — Diffusion synthetic for sonar imagery; basis for §2 alternative (e) deferral to Phase 2.
- Stabilizing underwater acoustic data generation with GAN (AS-DCGAN). *Intelligent Marine Technology and Systems*. DOI: 10.1007/s44295-025-00059-2. — Cautionary anchor for §2 alternative (e); GAN gains modest and brittle.

## 7. Decisions (C1 implementation spec)

1. **Module location:** `src/fathom/synthetic/` with submodules `ambient.py`, `biologicals.py`, `propagation.py`, `tonals.py`, `generator.py` (top-level orchestrator). Platform-layer per PCD v3 §7.4.
2. **Layered noise model:** NOAA ambient base + Watkins biologicals overlay + tonal injection + propagation modulation, in that order.
3. **NOAA curation:** 4-5 stations spanning Pacific/Atlantic/Arctic/tropical; ~10 hours per station; resampled to 32 kHz at load. Specific station IDs locked in C1 implementation; audit sidecar records IDs.
4. **Watkins curation:** WhaleNet recipe — drop <50-sample classes, dedupe, resample to median 47.6 kHz, stratified split. Specific class selection reviewed by CEO at C1 close.
5. **Tonal injection:** decaying-cosine pulses per §3.3 parameterization table. SNR computed against local ambient at tonal frequency.
6. **Propagation:** KRAKEN normal-mode for 3–500 Hz; BELLHOP ray-theory for >500 Hz. 5 canonical environments at C1 (deep isovelocity / downward-refracting summer / upward-refracting winter / shallow sediment-dominated / convergence-zone). Pre-computed impulse responses; per-injection sampling.
7. **Output:** waveform default at 32 kHz; optional spectrogram-direct path for training throughput. Audit sidecar on every synthetic gram with seed, source IDs, environment ID, full tonal parameter snapshot, KRAKEN/BELLHOP environment hash.
8. **Reproducibility:** seeded numpy RNG; manifest hash over synthetic dataset; SHA256 sidecars per file matching Sprint 1 audit pattern.
9. **PCD v3 §7.4 amendment** required: BELLHOP-only → KRAKEN/BELLHOP hybrid. Out-of-band CEO action.
10. **GAN/diffusion synthesis:** explicitly deferred to Phase 2.

## 8. Sign-off

**Drafted:** 2026-05-09 (Claude).
**CEO sign-off pending.** Append below upon review:

```
[CEO sign-off note here]
Date:
Decisions locked: §7 items 1-10 [accepted | revised: ...]
Comments:
```

After sign-off, this memo becomes the binding spec for Sprint 4 Cluster B1 (spike) and C1 (full implementation). Subsequent revisions go to a v2 memo, not in-place edits.
