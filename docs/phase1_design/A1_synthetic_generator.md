# Design Memo A1: Synthetic LOFAR Data Generator

**Status:** Signed off (revisions per `Design_Memo_Revision_Delta.md` 2026-05-09 incorporated). Binding spec for Sprint 4 Cluster B1 (spike) and C1 (full implementation).
**Author:** Claude (drafted from literature digest dispatched 2026-05-09; revised per CTO + external review feedback delta dated 2026-05-09).
**Scope:** Sprint 4 Cluster C1 implementation spec (`src/fathom/synthetic/`).
**Predecessor:** PCD v3 §7.4, Phase1_Plan.md, Sprint4_Plan.md
**Successor:** Sprint 4 Cluster B1 (spike, minimal viable generator) and C1 (full implementation) execute against this memo's §7 Decisions.

---

## 1. Problem statement

Tuor's Phase 1 ML line-detection model trains on a synthetic + real LOFAR data mix. Real data is severely limited (63 DeepShip vessels + 4 ShipsEar smoke recordings on disk; full-release acquisition stalled per Phase0_Review §6 item 1). Without realistic synthetic data, the ML detector either overfits to the small real distribution or underperforms classical at the same operating point — defeating the Phase 1 thesis that calibrated ML is the structural moat over classical-at-fixed-thresholds.

Per PCD v3 §7.4, the synthetic generator is platform-layer infrastructure (Fathom, not Tuor — future products consume the same generator). Two named risks bind the design:

- **Sim-to-real gap.** Clean synthetic noise produces models that fail on real ocean ambient. Mitigation per PCD v3 §7.4: "synthetic noise is modeled as realistic ocean noise; white noise is not used."
- **Synthetic-to-real ratio overfitting.** Too much synthetic in training causes overfitting to synthetic distribution. Mitigation per PCD v3 §7.4: "training mix is monitored as an explicit hyperparameter; ablations characterize the tradeoff."

The literature digest dispatched for this memo (Peng et al. 2025; Maddukuri et al. 2025; Liu et al. 2023; Synthio 2024; RadSimReal 2024; Haver et al. 2018; Licciardi & Carbone 2024; Bossér et al. 2024) surfaced both convergent practice and consequential debates. This memo decides the architecture; A3's memo handles evaluation methodology.

### Ground-truth labeling scope

Synthetic data provides exact line-level truth (per the §3.3.1 truth manifest schema). DeepShip provides real acoustic distribution and vessel-level labels, but NOT line-level annotations. Real line-detection evaluation therefore requires one of:

(a) manual line annotations by a qualified operator on a small panel,
(b) synthetic injection of known lines into real ambient clips, or
(c) a clearly marked weak-label protocol (e.g., classical-detector pseudo-labels at a conservative operating point, explicitly flagged as noisy ground truth).

Phase 1 uses (b) as the primary real-evaluation method (Tier 2 in A3 §3.1). Phase 2 adds (a) on self-collected data.

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

**Frequency-range limitation:** NOAA NRS is calibrated 10-2000 Hz. The LOFAR primary view extends to 3 Hz (blade-rate fundamentals at 5-12 Hz). Phase 1 evaluation scope is **10-1000 Hz for the primary gate.** Content below 10 Hz is present in DeepShip/ShipsEar recordings but not modeled by the NOAA ambient layer. Blade-rate signatures remain detectable via harmonics above 10 Hz in most cases. If Phase 2 evaluation requires explicit 3-10 Hz ambient modeling, a separate low-frequency ambient source (e.g., DeepShip-derived ambient characterization from vessel-free segments) will be added.

Manifest sidecar lists the specific station IDs, recording timestamps, and SHA256 hashes of each curated source clip. Audit-trail on every synthetic gram identifies which NOAA source was the base layer.

### 3.2 Layer 2 — Watkins biological transients (distractors)

Watkins Marine Mammal Sound Database (https://cis.whoi.edu/science/B/whalesounds/) overlaid as distractors at controlled rate. Curation per the WhaleNet recipe (Licciardi & Carbone 2024, *IEEE Access*):

- **Drop classes with <50 samples** (51 → ~32 species).
- **De-duplicate** (catalog has known duplicates).
- **Resample to median rate (47.6 kHz)**, not the floor — preserves high-frequency content.
- **Stratified split** before training so biological classes are seen in both train and val.

**Scope limitation:** Watkins is primarily a marine-mammal database. The <50 sample threshold may eliminate non-mammal confusers (snapping shrimp, fish chorusing) that are operationally relevant false-alarm sources. If CEO review at C1 close identifies missing non-mammal confusers, either relax the threshold for selected confuser classes or add a supplementary biological-noise source in Sprint 5.

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

### 3.3.1 Synthetic truth manifest

Every generated clip carries a machine-readable truth manifest (JSON sidecar, separate from the audit/provenance sidecar) with per-line ground truth.

Schema (per injected line):

```json
{
  "line_id": "str",                      // unique within clip
  "source_type": "tonal",                // future: "biological", "broadband"
  "harmonic_id": 0,                      // 0=fundamental, 1=first harmonic, etc.
  "f0_hz": 50.0,                         // fundamental frequency
  "freq_curve_hz": [50.0, 50.1, ...],    // frequency at each STFT frame (captures drift)
  "t_start_s": 1.5,                      // onset time in clip
  "t_end_s": 47.3,                       // offset time in clip
  "snr_curve_db": [12.4, 12.5, ...],     // per-frame SNR against local ambient
  "persistence_s": 45.8,                 // total active duration
  "drift_rate_hz_per_s": 0.02,           // mean drift rate
  "mask_bin_indices": [[t_frame, f_bin], ...],  // exact mask cells
  "generation_seed": 20260520
}
```

Clip-level manifest additionally contains:
- `negative_label`: bool (true if clip contains zero injected lines)
- `confuser_labels`: list of `{species, watkins_id, t_start, t_end, freq_range}`
- `noaa_source_id`, `noaa_clip_timestamp`
- `propagation_environment_id` (which canonical IR was applied)
- `generator_version`

This manifest is what A2 trains against and A3 evaluates against. The audit sidecar (Sprint 1 pattern) tracks provenance; the truth manifest tracks labels. **These are separate files with separate purposes.**

### 3.4 Layer 4 — Propagation artifacts (pre-computed KRAKEN/BELLHOP IRs)

Hybrid propagation modeler using **PRE-COMPUTED impulse responses (not runtime KRAKEN/BELLHOP execution per sample):**

- **3–500 Hz: KRAKEN normal-mode impulse responses.** Pre-computed for **5 canonical environments × 10 source-receiver geometry combinations = 50 IRs total.** Per-injection: sample environment uniformly, sample geometry uniformly, convolve tonal waveform with selected IR. No per-sample KRAKEN runtime.
- **>500 Hz: BELLHOP ray-theory impulse responses.** Same 5 environments × 10 geometries = 50 IRs. Cached at C1 implementation time.

Canonical environments: deep ocean isovelocity, downward-refracting summer profile, upward-refracting winter profile, shallow-water sediment-dominated, convergence-zone. Source-receiver geometries: ranges 1/5/10/20/50 km × source depths 50/200 m (covers surface vessels through patrol-depth submarines).

Propagation is applied to the injected tonal *before* mixing with ambient. This gives realistic multipath striations, shadow-zone effects, and convergence-zone gain variation that operators read as cues.

This is the **PCD v3 §7.4 amendment Keith approved out-of-band** — original PCD v3 §7.4 said "Bellhop-derived"; revised to "pre-computed KRAKEN low-freq + BELLHOP high-freq IRs."

### 3.5 Mixing and rendering

Final waveform = ambient + biologicals + propagation-applied tonal.

Output options:
- **Waveform output** for full-pipeline-realism training (resampled to 32 kHz).
- **Direct LOFAR-spectrogram output** (skip waveform reconstruction) for training-throughput optimization. C1 implementation defaults to waveform; spectrogram-direct is an optional optimization if training throughput becomes binding.

Reproducibility: all randomness via seeded `numpy.random.default_rng(seed)`. Each synthetic gram carries an audit sidecar with seed, NOAA source ID + clip timestamp, Watkins selection IDs, KRAKEN/BELLHOP environment ID, and full tonal-injection parameter snapshot. **Truth manifest sidecar (separate file)** carries the per-line ground truth per §3.3.1. Manifest hash over the synthetic dataset locks reproducibility.

## 4. Evaluation methodology (preview)

Full evaluation methodology lives in `A3_sim_to_real_evaluation.md`. Three-tier framework summarized:

- **Tier 1 — Synthetic exact-truth test.** Held-out synthetic test set; evaluates whether the model learns the generator's distribution. Sprint 4 C3 acceptance gate.
- **Tier 2 — Real-ambient injection test.** Inject known synthetic tonals into held-out real DeepShip ambient; primary real-evaluation method for the ratio sweep.
- **Tier 3 — Operator-labeled natural-real panel.** Sprint 6+; small CEO-annotated panel; qualitative.

Ratio sweep operates on Tier 2. The synthetic generator iteration loop terminates when the Tier 1 sanity check passes and Tier 2 ratio sweep finds a U-minimum that beats real-only training.

## 5. Risks and mitigations

1. **KRAKEN parameter sensitivity.** Normal-mode propagation depends on water-depth profile, sound-speed profile, bottom type. Picking a fixed set of canonical environments vs. varying continuously is a cost-vs-realism tradeoff. **Mitigation:** start with 5 environments × 10 geometries = 50 IRs; if A3 evaluation surfaces overfitting to a single environment regime, expand to 10-15 environments in Sprint 5.
2. **Watkins biological-class match.** Choosing biologicals that match real-world false-alarm sources is a domain-knowledge decision. **Mitigation:** CEO reviews specific Watkins selections at C1 close against IUSS operator memory of what biological false alarms actually look like; iterate if mismatch surfaces. Non-mammal confusers (snapping shrimp, fish chorusing) may need supplementary source per §3.2 scope limitation.
3. **NOAA archive curation cost.** Full archive is 300+ TB; we need a curated subset. **Mitigation:** start with 4-5 stations covering different ambient regimes (Pacific/Atlantic/Arctic/tropical); ~10 hours per station ≈ 50 hours total ≈ 200 GB. Expand if A3 evaluation shows overfitting to a single regime.
4. **KRAKEN library availability and licensing.** Open-source KRAKEN implementations exist (`pyat`, `arlpy`) but are research-grade. **Mitigation:** §3.4 spec explicitly uses pre-computed IRs at C1 implementation time; no per-sample KRAKEN runtime. Falls back to canonical pre-computed IRs from established acoustic textbooks if `pyat`/`arlpy` integration is fragile.
5. **Drift-rate calibration.** Few published synthetic generators model drift; our Gaussian σ=0.05 Hz/s prior is an educated guess. **Mitigation:** A3 ratio-sweep ablation will surface whether drift modeling is helping or hurting; tune the prior in Sprint 5 if needed.
6. **Defer-GAN regret.** If KRAKEN+sample-based synthetic plateaus at sim-to-real gap larger than acceptable, GAN/diffusion is the Phase 2 lever. **Mitigation:** A3 evaluation's primary gate will surface the plateau cleanly; Phase 2 plan inherits a clear "GAN augmentation" cluster scoping if needed.
7. **Frequency-range limitation (10-1000 Hz).** Below-10-Hz blade-rate fundamentals (3-10 Hz) are not modeled by NOAA NRS. **Mitigation:** Phase 1 detection relies on harmonics ≥10 Hz; PCD v3 §6.6 commits to 1-1000 Hz primary view but Phase 1 evaluation is scoped to 10-1000 Hz. Phase 2 adds explicit 3-10 Hz modeling if operator review flags the gap.

## 6. References

The literature digest sourcing this memo cited:

- Peng, D., Xu, X., Song, W., Gao, D. (2025). "Preprocessing LOFARgram through U-Net++ neural network." *Frontiers in Marine Science* 12:1528111. DOI: 10.3389/fmars.2025.1528111. — KRAKEN normal-mode + decaying-cosine source primitives; primary literature anchor for §3.3 and §3.4.
- Haver, S. M. et al. (2018). "Monitoring long-term soundscape trends in U.S. waters: The NOAA/NPS Ocean Noise Reference Station Network." *Marine Policy*. DOI: 10.1016/j.marpol.2017.11.024. — NOAA NRS canonical reference for §3.1.
- Licciardi, A., & Carbone, D. (2024). "WhaleNet: A Novel Deep Learning Architecture for Marine Mammals Vocalizations on Watkins Marine Mammal Sound Database." *IEEE Access* 12. arXiv:2402.17775. — Watkins curation recipe for §3.2.
- Maddukuri, S. et al. (2025). "The Science of Co-Training: Sim-and-Real for Robot Manipulation." — Synthetic-to-real ratio findings (~38% optimal, U-shaped degradation, structured alignment dominant); informed §4 ratio sweep.
- Liu, Y., Yi, T. et al. (2023). "Data augmentation method for underwater acoustic target recognition based on underwater acoustic channel modeling and transfer learning." *Applied Acoustics* 211. DOI: 10.1016/j.apacoust.2023.109552.
- Bossér, D., Nordenvaad, M.L., Hendeby, G., & Skog, I. (2024). "Broadband Passive Sonar Track-Before-Detect Using Raw Acoustic Data." arXiv:2412.15727. — Vector-autoregressive ambient-noise modeling validates non-Gaussian ambient.
- Synthio: Ghosh, S. et al. (2025). "Augmenting Small-Scale Audio Classification Datasets with Synthetic Data." ICLR 2025. arXiv:2410.02056.
- Agrawal, A. et al. (2024). "Syn2Real Domain Generalization for Underwater Mine-like Object Detection Using Side-Scan Sonar." arXiv:2410.12953.
- Stabilizing underwater acoustic data generation with GAN (AS-DCGAN). *Intelligent Marine Technology and Systems*. DOI: 10.1007/s44295-025-00059-2.

## 7. Decisions (C1 implementation spec)

1. **Module location:** `src/fathom/synthetic/` with submodules `ambient.py`, `biologicals.py`, `propagation.py`, `tonals.py`, `generator.py` (top-level orchestrator). Platform-layer per PCD v3 §7.4.
2. **Layered noise model:** NOAA ambient base + Watkins biologicals overlay + tonal injection + propagation modulation, in that order.
3. **NOAA curation:** 4-5 stations spanning Pacific/Atlantic/Arctic/tropical; ~10 hours per station; resampled to 32 kHz at load. Specific station IDs locked in C1 implementation; audit sidecar records IDs.
4. **Watkins curation:** WhaleNet recipe — drop <50-sample classes, dedupe, resample to median 47.6 kHz, stratified split. Specific class selection reviewed by CEO at C1 close. Non-mammal confuser supplement deferred to Sprint 5 if needed.
5. **Tonal injection:** decaying-cosine pulses per §3.3 parameterization table. SNR computed against local ambient at tonal frequency.
6. **Propagation:** Pre-computed KRAKEN normal-mode IRs for 3–500 Hz; pre-computed BELLHOP ray-theory IRs for >500 Hz. 5 canonical environments × 10 source-receiver geometries = 50 IRs per propagation regime. **No per-sample runtime KRAKEN/BELLHOP.** Expand to 10-15 environments in Sprint 5 if A3 evaluation surfaces environment-overfitting.
7. **Output:** waveform default at 32 kHz; optional spectrogram-direct path for training throughput. Audit sidecar on every synthetic gram with seed, source IDs, environment ID, full tonal parameter snapshot, KRAKEN/BELLHOP environment hash.
8. **Reproducibility:** seeded numpy RNG; manifest hash over synthetic dataset; SHA256 sidecars per file matching Sprint 1 audit pattern.
9. **PCD v3 §7.4 amendment** required: BELLHOP-only → pre-computed KRAKEN/BELLHOP hybrid. Out-of-band CEO action.
10. **Truth manifest:** every synthetic clip carries a JSON truth-manifest sidecar per the §3.3.1 schema. This is the training label source for A2 and the evaluation ground truth for A3 Tier 1. Separate from the audit/provenance sidecar.
11. **Frequency-range scope:** Phase 1 evaluation operates on 10-1000 Hz. Below-10-Hz ambient is not modeled by NOAA NRS; blade-rate detection relies on harmonics above 10 Hz. Explicit 3-10 Hz modeling deferred to Phase 2.
12. **Dataset licensing:** All external datasets (NOAA NRS, Watkins, DeepShip, ShipsEar) must be cleared for intended use (internal R&D / government prototype) before integration. NOAA is public domain (federal). Watkins and DeepShip require license review at C1 start. If any source restricts commercial/defense use, substitute or restrict to internal-only with explicit documentation.
13. **Staged implementation:** Sprint 4 B1 spike implements the **minimal viable generator** (NOAA ambient + deterministic tonal injection + truth manifest, NO propagation, NO biologicals). Sprint 4 C1 adds the full layered model (biologicals, KRAKEN/BELLHOP IRs, drift). This staging proves the training pipeline works on simple synthetic before adding realism layers.
14. **GAN/diffusion synthesis:** explicitly deferred to Phase 2.

## 8. Sign-off

**Drafted:** 2026-05-09 (Claude). **Revised:** 2026-05-09 (per `Design_Memo_Revision_Delta.md` 2026-05-09).

```
CEO sign-off:
Date: 2026-05-09
Technical direction accepted. Revisions from Design_Memo_Revision_Delta
(2026-05-09) incorporated.
Decisions locked: §7 items 1-14 [accepted]
PCD v3 §7.4 amendment: BELLHOP-only -> KRAKEN/BELLHOP hybrid (pre-computed
IRs). Evaluation scope 10-1000 Hz for Phase 1.
```


---

## Addendum: 2026-05-13 — Sprint 4 close-out deltas

**§3.4 propagation — C1.3-lite substitution (in production).** A1 §3.4 specified a pre-computed KRAKEN/BELLHOP IR library (5 envs × 10 geometries × 2 bands = 100 IRs). Pre-step 0 canonical-IR hunt 2026-05-12 confirmed no public 3-1000 Hz IR dataset exists (every measured-IR library targets underwater comms in kHz range). C1.3-lite substitutes a parametric three-path channel (direct + surface bounce + bottom bounce) + Thorpe absorption with geometry sampled from priors. Implementation: `src/fathom/synthetic/propagation.py`. Four documented A1 §3.4 deltas land in every audit sidecar. SWellEx-96 (UCSD MPL) flagged as the sole public dataset where sim-to-real CIR validation is possible via Nannuru SBL extraction; deferred to Sprint 5+ as a candidate validation-loop cluster.

**§3.3 priors — Sprint 5 widening candidates.** C4 (2026-05-13) on a real DeepShip Tug recording: U-Net trained exclusively on synthetic data found only the two strongest persistent tonals (384.8 Hz, 443.4 Hz) and missed the dozens of drifting/broadband features that fill the 50-400 Hz range. Attributed to synthetic training distribution not covering real tug machinery content. Sprint 5 candidates for prior widening (to be ratified during Sprint 5 design):
- `drift_rate_std_hz_per_s`: 0.05 → wider (1.0+) to cover variable-load tugs.
- `n_harmonics_choices`: (1, 2, 3) → wider (1..6) to cover richer harmonic stacks.
- `harmonic_decay_range`: revisit so weak harmonics survive at higher n_harmonics.
- `total_persistence_log_range`: skewed toward longer persistence to capture sustained machinery tonals.

**§3.3.1 truth manifest — propagation fields added.** `SyntheticLineGroundTruth` now carries optional `propagation_geometry: SyntheticPropagationGeometry | None` + `propagation_model_id: str | None` (set when C1.3-lite is enabled, else None). The Pydantic `SyntheticPropagationGeometry` mirrors the dataclass `SampledPropagationGeometry` in `src/fathom/synthetic/priors.py`.
