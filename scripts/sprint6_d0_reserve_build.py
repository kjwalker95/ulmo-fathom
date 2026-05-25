"""Sprint 6 Cluster D.0 — Tier-3 reserve calibration dataset build.

Mirrors scripts/sprint6_aprime_build.py but targets the test partition's
tier3_reserve subset (Cargo/103, Cargo/69, Tanker/13 - the 3 viable
reserve vessels at >=35s; Passengership/6, Tanker/10, Tanker/35 are
below the 33.15s patch-extraction hard floor and get min-duration-
skipped at build time).

Expected output: 3 vessels x 10 clips = 30 clips, ~180 patches after
extraction (Sprint 5 effective ratio ~6 patches/clip).

The seed (20260524) is fresh from A's 20260515 / A's 20260522 / B's
20260601-20260605 so the clip-variation RNG doesn't collide with any
prior dataset.
"""
from pathlib import Path

from fathom.evaluation.tier2 import build_tier2_dataset
from fathom.synthetic.priors import TonalParameterPriors


def main() -> None:
    manifest = build_tier2_dataset(
        split_manifest_path=Path("artifacts/sprint3_splits/deepship_splits.json"),
        partition="test",
        deepship_root=Path("/Users/keith/Documents/data/DeepShip"),
        n_clips_per_vessel=10,
        seed=20260524,
        tonal_priors=TonalParameterPriors(),
        out_dir=Path("data/tier2_test_v2_reserve_10c"),
        vessel_subset_path=Path("artifacts/sprint5_test_partition.json"),
        subset_key="tier3_reserve",
        clip_duration_s=40.0,
        target_sample_rate=32_000,
        min_duration_s=35.0,
    )
    print(
        f"vessels: {manifest.n_vessels}  clips: {manifest.n_clips_total}  "
        f"seed: {manifest.seed}  out: {Path('data/tier2_test_v2_reserve_10c').resolve()}"
    )


if __name__ == "__main__":
    main()