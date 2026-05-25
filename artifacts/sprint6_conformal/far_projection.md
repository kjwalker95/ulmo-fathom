# Sprint 6 D.6 - 360-beam FAR projection

- Per the C.4 winner: max_mean confidence + ensemble of 5 U-Nets
- Calibration set: Tier-3 reserve (n_cal=180, pos=94, neg=86)
- Per-class bounds: positive 0.1031, negative 0.1078
- Val evaluation set: 300 patches, 66 negative
- Patches/hour/beam (derived from default_lofar_config): 219.7
- alpha=0.05 fit + plotted but NOT committed for FAR per the D.0 thin-negative-class triage; only alpha in {0.10, 0.20} committed.

| alpha | p_FP | per-beam FAR/hr | 180-beam alerts/hr | <0.05/hr? |
|---:|---:|---:|---:|:---:|
| 0.10 | 0.4242 | 93.2173 | 16779.12 | MISS |
| 0.20 | 0.1515 | 33.2919 | 5992.54 | MISS |

Watch Supervisor tolerance: 5-10 alerts/hr aggregate.
