"""Tier-2 round-trip integration test (Sprint 5 Cluster C1).

Picks the first val-partition DeepShip recording (post-A0 compound key),
injects 3 known tonals at controlled SNRs ({2, 8, 15} dB), builds a
SyntheticPatchDataset over the single-clip output, loads the Sprint 4
baseline U-Net checkpoint, runs evaluate_model. Pass condition: metrics
dict is well-formed.

Acceptance is plumbing-level. Sprint 4 U-Net has a known sim-to-real recall
gap on real ambient; that's Cluster C2-C4's problem to fix.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import torch
from rich.console import Console

from fathom.detection.ml_data import (
    PatchExtractionConfig,
    SyntheticPatchDataset,
    default_lofar_config,
)
from fathom.detection.ml_eval import evaluate_model, print_eval_summary
from fathom.detection.ml_train import build_model
from fathom.evaluation import inject_into_real_ambient
from fathom.synthetic.priors import SampledTonalParameters

CONSOLE = Console()


def _resolve_first_val_ambient(
    deepship_root: Path, split_manifest_path: Path
) -> Path:
    splits = json.loads(split_manifest_path.read_text())
    val_keys = splits["val_vessels"]
    if not val_keys:
        raise click.UsageError("split manifest has no val vessels")
    compound = val_keys[0]
    class_label, stem = compound.split("/", 1)
    wav = deepship_root / class_label / f"{stem}.wav"
    if not wav.exists():
        raise click.UsageError(f"resolved val ambient not found: {wav}")
    return wav


@click.command()
@click.option(
    "--deepship-root",
    type=click.Path(exists=True, path_type=Path),
    default=Path("/Users/keith/Documents/data/DeepShip"),
)
@click.option(
    "--split-manifest",
    type=click.Path(exists=True, path_type=Path),
    default=Path("artifacts/sprint3_splits/deepship_splits.json"),
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    default=Path("artifacts/sprint4_baseline/unet_seed20260512/best.pt"),
)
@click.option(
    "--unet-base-channels", type=int, default=32,
    help="Sprint 4 baseline is base_channels=32 (MPS-bound). "
    "Pass 64 for A100 checkpoints.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("artifacts/sprint5_c1_smoke"),
)
@click.option("--seed", type=int, default=20260513)
@click.option(
    "--device", type=str, default="cpu",
    help="cpu / mps / cuda. Single-clip smoke; CPU is fine.",
)
@click.option(
    "--clip-duration-s", type=float, default=40.0,
    help="Clip duration cap (s). Matches v1 ~38 sec gram duration.",
)
def main(
    deepship_root: Path, split_manifest: Path, checkpoint: Path,
    unet_base_channels: int, out_dir: Path, seed: int, device: str,
    clip_duration_s: float,
):
    ambient = _resolve_first_val_ambient(deepship_root, split_manifest)
    CONSOLE.print(f"[cyan]Tier-2 smoke ambient: {ambient}[/cyan]")

    explicit_params = [
        SampledTonalParameters(
            f0_hz=80.0, n_harmonics=3, harmonic_decay=0.6,
            decay_constant_per_s=1.0, cluster_period_s=5.0,
            total_persistence_s=20.0, drift_rate_hz_per_s=0.0,
            target_snr_db=2.0, t_onset_s=0.0,
        ),
        SampledTonalParameters(
            f0_hz=200.0, n_harmonics=3, harmonic_decay=0.6,
            decay_constant_per_s=1.0, cluster_period_s=5.0,
            total_persistence_s=20.0, drift_rate_hz_per_s=0.0,
            target_snr_db=8.0, t_onset_s=0.0,
        ),
        SampledTonalParameters(
            f0_hz=350.0, n_harmonics=3, harmonic_decay=0.6,
            decay_constant_per_s=1.0, cluster_period_s=5.0,
            total_persistence_s=20.0, drift_rate_hz_per_s=0.0,
            target_snr_db=15.0, t_onset_s=0.0,
        ),
    ]

    result = inject_into_real_ambient(
        ambient_path=ambient,
        out_dir=out_dir,
        clip_id="c1_smoke",
        seed=seed,
        explicit_params=explicit_params,
        clip_duration_s=clip_duration_s,
    )
    CONSOLE.print(
        f"[cyan]Injected {result.n_sources_realized} sources -> "
        f"{len(result.manifest.lines)} truth-line rows[/cyan]"
    )

    patch_config = PatchExtractionConfig(target_mode="mask")
    dataset = SyntheticPatchDataset(
        clip_paths=[result.wav_path],
        lofar_config=default_lofar_config(),
        patch_config=patch_config,
    )
    CONSOLE.print(f"[cyan]Dataset patches: {len(dataset)}[/cyan]")

    dev = torch.device(device)
    model = build_model(
        architecture="unet",
        unet_base_channels=unet_base_channels,
    ).to(dev)
    state = torch.load(str(checkpoint), map_location=dev)
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    CONSOLE.print(f"[cyan]Loaded {checkpoint.name}[/cyan]")

    metrics = evaluate_model(
        model=model,
        dataset=dataset,
        device=dev,
        architecture="unet",
        class_threshold=0.5,
        bin_threshold=0.3,
        iou_threshold=0.1,
    )
    print_eval_summary(metrics, console=CONSOLE)

    if not isinstance(metrics, dict) or not metrics:
        raise click.UsageError("evaluate_model returned empty metrics; plumbing failure")
    CONSOLE.print("[green]Tier-2 round-trip smoke PASSED[/green]")


if __name__ == "__main__":
    main()
