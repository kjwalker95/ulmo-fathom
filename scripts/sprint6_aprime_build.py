"""Sprint 6 A' — Tier-2 train corpus expansion to 10 clips/vessel.

Generates data/tier2_train_v2_10c/ from the train partition of the post-A0
split manifest. Uses the same priors as Sprint 5's tier2_train_v2 baseline
(default TonalParameterPriors); only n_clips_per_vessel changes from 1 to 10.

Expected output: 43 train vessels × 10 clips = 430 attempts. Any vessel with
ambient < 35 s gets warn-and-skipped by build_tier2_dataset's A.4 min-duration
filter; Sprint 5 saw no train-partition skips so expect 430.
"""
from pathlib import Path

from fathom.evaluation.tier2 import build_tier2_dataset
from fathom.synthetic.priors import TonalParameterPriors


def main() -> None:
    manifest = build_tier2_dataset(
        split_manifest_path=Path("artifacts/sprint3_splits/deepship_splits.json"),
        partition="train",
        deepship_root=Path("/Users/keith/Documents/data/DeepShip"),
        n_clips_per_vessel=10,
        seed=20260522,
        tonal_priors=TonalParameterPriors(),
        out_dir=Path("data/tier2_train_v2_10c"),
        clip_duration_s=40.0,
        target_sample_rate=32_000,
        min_duration_s=35.0,
    )
    print(
        f"vessels: {manifest.n_vessels}  clips: {manifest.n_clips_total}  "
        f"seed: {manifest.seed}  out: {Path('data/tier2_train_v2_10c').resolve()}"
    )


if __name__ == "__main__":
    main()