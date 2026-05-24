# Sprint 6 C.2 reliability summary

- Val: data/tier2_val_v2  (300 patches, 234 positive)
- Sprint 5 C5 single-model baseline: ECE = 0.0746 +/- 0.0101 (2-3 occupied bins)

| Setup | ECE | Occupied bins / 10 | Overconfidence frac |
|---|---:|---:|---:|
| member_seed20260601_max_mean | 0.0866 | 4 | 0.75 |
| member_seed20260602_max_mean | 0.0795 | 2 | 0.50 |
| member_seed20260603_max_mean | 0.0836 | 3 | 0.67 |
| member_seed20260604_max_mean | 0.0798 | 2 | 0.50 |
| member_seed20260605_max_mean | 0.0661 | 2 | 0.50 |
| ensemble_max_mean | 0.0544 | 6 | 0.67 |
| ensemble_mean_max | 0.0610 | 7 | 0.57 |
| ensemble_peak_freq_band | 0.1163 | 10 | 0.10 |

C.4 acceptance: at least one setup should produce >=5 occupied bins AND ECE < 0.0746. Winning (function, aggregation) pair feeds Cluster D.
