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

(Empty as of Sprint 1 mid-execution. Populate after operator review.)
