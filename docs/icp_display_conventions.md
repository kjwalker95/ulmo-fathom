# ICP display conventions (initial capture)

This doc captures what we know about the Lockheed Martin Integrated Common Processor (ICP) gram display conventions that Fathom's LOFAR rendering should match as closely as possible. It is the iteration substrate for the end-of-Sprint-1 operator review.

The CEO stood watch on IUSS for four years; this doc is the union of what he can recall plus what we infer from the published acoustic-analysis literature. Display conventions may have evolved on the ICP since; the operator review at end of Sprint 1 is the first round of correction.

## Known conventions (CEO operator memory)

### Frequency axis
- **Linear**, not logarithmic, not mel-scale. PCD v2 §6.2 commits to linear; mel-scale perceptually weights frequencies for human auditory perception which is the wrong choice for tonal-line analysis at 5-12 Hz.
- Primary view: 1-1000 Hz where submarine vulnerability features live (blade-rate harmonics 5-12 Hz, auxiliary-machinery tonals near 50 Hz).
- Wider view (1-5000 Hz) available for higher-speed contacts and surface-vessel context.

### Time axis
- Z-time (Zulu / UTC) labels. Operators report contacts as "line of interest on array XXXX, beam XXXX, at HHMM Zulu."
- Time scale varies; longer recordings span minutes-to-hours per gram column for steady-state signature observation; shorter spans for tactical contacts.

### Intensity / colormap
- The historical ICP defaults trend toward grayscale-with-emphasis or red-orange palettes. Modern ICP refresh cycles may have shifted defaults; operator review confirms.
- Dynamic range: enough that ambient is visible but tonal lines stand out clearly — typically 40-60 dB shown.
- Normalization: ambient-subtracted (per-frequency-bin local-ambient estimation) so tonal lines pop against background rather than getting lost in broadband variation.

### Reading conventions
- Tonal lines appear as horizontal stripes (constant frequency, persistent in time).
- Frequency-modulated tonals appear as slightly tilted or wavy lines.
- Biological transients (whale calls, snapping shrimp) appear as short blobs or harmonics that don't persist.
- Operators look for persistence: a tonal that holds for tens of seconds to minutes is a contact candidate.

## Sprint 1 default rendering choices

The first-pass values in `configs/sprint1.yaml` are:
- `colormap: viridis` — a perceptually uniform default that doesn't match historical ICP defaults but is operator-readable. **Iterate at operator review.**
- `intensity_dynamic_range_db: 50` — within typical operator range.
- `figure_size_in: [12, 8]`, `dpi: 120` — reasonable for a 16:9 inch screen at typical reading distance.
- LOFAR n_fft 16384 at 32 kHz → ~1.95 Hz frequency bins, which resolves the 5-12 Hz blade-rate band.

## Open questions for operator review

These are the questions to resolve at end of Sprint 1 by walking through `artifacts/sprint1_sanity/INDEX.md` with the CEO. Capture answers here so they don't get lost.

- [ ] Does `viridis` match operator intuition, or do we need a closer-to-ICP palette (greyscale, red-orange ramp, or custom)?
- [ ] Is the dynamic range (50 dB) too wide / too narrow? What range do the ICP defaults use?
- [ ] Does the time-axis labeling format read correctly to an operator? Should time-zero anchor at recording start or at file UTC timestamp?
- [ ] Does the frequency-axis label format read correctly? Linear ticks at sensible round numbers?
- [ ] Should we ship two grams per recording (1-1000 Hz primary view + 1-5000 Hz wide view) by default, or one with toggling?
- [ ] Is the 75% STFT overlap (hop_length 4096 at n_fft 16384, ~128 ms time bins) too coarse / too fine for operator pattern matching?
- [ ] Are there ICP display conventions for line-of-interest overlays (color, line style, label placement) that Sprint 2's detection overlays should match?
- [ ] What is the ICP's normalization approach exactly? Two-Pass Split Window or a different ambient estimator? (Affects Sprint 2 line-detection design.)

## Resolution log

**Sprint 1 operator review (2026-05-06).** No parameter changes required; defaults from `configs/sprint1.yaml` confirmed against operator intuition on DeepShip recordings.

- LOFAR `n_fft=16384` at 32 kHz → ~1.95 Hz frequency resolution; resolves the 5–12 Hz blade-rate vulnerability band.
- Frequency range: 1–1000 Hz primary view (where submarine vulnerability features live). Wide view (1–5000 Hz) available for higher-speed contacts.
- Color map: `viridis` at 50 dB dynamic range. Operator-readable; ICP-exact convention question stays open until current/recently-retired-operator engagement opens that channel via NUWC / CRADA.
- Split-window normalization at `train=33 / central=5 / gap=1` produces operator-credible grams on the first pass without iteration.

See `Sprint1_Retro.md` for the full retro and the eight-cluster ship list.

The remaining open questions in this document (ICP-exact color/scale conventions, line-overlay style, normalization approach used on the ICP itself) carry forward to be resolved through operator engagement. Sprint 2 line detection ships against the Sprint-1-confirmed display substrate.

**Display orientation correction (2026-05-10).** Sprint 1 implemented Convention A (academic "frequency-vs-time" plot — time horizontal, freq vertical, tonal lines horizontal). CEO surfaced operational memory while reviewing B1 spike: IUSS LOFAR display is **Convention B** (waterfall — frequency horizontal, time vertical, tonal lines vertical, "static-TV-screen-with-vertical-contact-bars" appearance, time flowing top-to-bottom with newest at bottom). Sprint 1 review confirmed parameters (n_fft, freq range, color, normalization) but did not anchor on axis orientation. Corrected in `src/fathom/display/render.py` 2026-05-10. Detection logic is orientation-agnostic — only the rendering layer changed.

**Colormap correction (2026-05-10).** Sprint 1 confirmed `viridis` as "operator-readable" but did not test against operational-convention grayscale. CEO surfaced operational reference image showing IUSS-standard grayscale halftone display (dark contact bars on light dotted background — "static-TV" appearance). Default colormap changed `viridis` → `Greys` (matplotlib forward grayscale, high values dark) in `RenderConfig` and propagated to `configs/sprint{1,2,3}.yaml` and `scripts/build_synthetic_b1.py`. The intermediate `Greys_r` (reverse grayscale, high values bright) was tried first and gave the wrong polarity; corrected to `Greys`. Existing viridis rendering retained as alternative-view option via `RenderConfig.colormap` override.

The split-window normalization produces a characteristic white "halo" immediately adjacent to strong tonals (the tonal contaminates the train-ring ambient estimate of nearby bins, biasing their residual low). This is expected and matches operational-LOFAR appearance.

Open question deferred to PCD v3 amendment: PCD v3 §2.2 reads "persistent horizontal lines that indicate a contact" — Convention A wording. Should be revised to "persistent vertical lines" per Convention B. CEO out-of-band action.

**Forward-looking (Phase 2):** software will need configurable view profiles — operational LOFAR (current default), analytical (viridis smooth), BTR (bearing-time), composite multi-panel (BTR + LOFAR side-by-side as in operational watch-floor displays). Backlog item; not Phase 1.
