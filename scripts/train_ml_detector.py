"""ML detector training entry point (C3 cluster).

C3.a (this version): scaffolding — argparse, clip-level train/val split,
SyntheticPatchDataset construction (target_mode per architecture),
WeightedRandomSampler for 50/50 batch balance, DataLoader. NO training loop.

C3.b adds augmentation. C3.c adds the train/eval loop with model + losses
+ optimizer + scheduler. C3.d adds checkpoint + metrics persistence.
C3.e adds the Tier-1 evaluation harness.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
from rich.console import Console
from torch.utils.data import DataLoader
from datetime import datetime, timezone

from fathom.detection.ml_persist import (
    MetricsLogger,
    TrainingRunMetadata,
    save_checkpoint,
    save_run_metadata,
)
from fathom.detection.ml_augment import PatchAugmentation
from fathom.detection.ml_data import (
    PatchExtractionConfig,
    SyntheticPatchDataset,
    default_lofar_config,
    make_balanced_patch_sampler,
)
from fathom.detection.ml_train import (
    build_loss,
    build_model,
    evaluate,
    train_one_epoch,
)
from fathom.detection.ml_eval import evaluate_model, print_eval_summary
CONSOLE = Console()


def _clip_level_train_val_split(
    clip_paths: list[Path], val_fraction: float, seed: int,
) -> tuple[list[Path], list[Path]]:
    """Split clip_paths into train/val at the CLIP level (not patch level)
    to prevent leakage from sibling patches of the same clip ending up in
    both splits.
    """
    rng = np.random.default_rng(seed)
    n_total = len(clip_paths)
    n_val = max(1, int(n_total * val_fraction))
    indices = rng.permutation(n_total)
    val_set = set(int(i) for i in indices[:n_val].tolist())
    train_paths = [p for i, p in enumerate(clip_paths) if i not in val_set]
    val_paths = [p for i, p in enumerate(clip_paths) if i in val_set]
    return train_paths, val_paths


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Bulk training dataset directory (output of build_training_dataset.py).",
)
@click.option(
    "--architecture",
    type=click.Choice(["resnet18", "unet"]),
    default="resnet18",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("artifacts/c3_training"),
)
@click.option("--batch-size", type=int, default=64)
@click.option("--epochs", type=int, default=2, help="2 for smoke; 50 for A2 baseline.")
@click.option("--val-fraction", type=float, default=0.2)
@click.option("--seed", type=int, default=20260512)
@click.option(
    "--n-samples-per-epoch",
    type=int,
    default=20000,
    help="WeightedRandomSampler num_samples (A2 baseline 20k).",
)
@click.option("--device", type=str, default="auto", help="auto|cpu|mps|cuda")
@click.option(
    "--unet-base-channels",
    type=int,
    default=64,
    help="U-Net base channels: 64 (31M params, default) or 32 (7.7M ablation per CLAUDE.md). "
         "Drop to 32 if MPS OOMs or for apples-to-apples vs ResNet-18 (11M).",
)
@click.option(
    "--num-workers",
    type=int,
    default=0,
    help="DataLoader num_workers. 0 = single-process (Sprint 4 default; "
         "preserves backward-compat). Set 4-8 for CUDA training; 16-24 on A100 "
         "to keep the GPU fed. Requires Cluster A2 pre-computed grams "
         "(.lofar.npz) to actually escape the CPU LOFAR-STFT bottleneck.",
)
@click.option(
    "--pin-memory/--no-pin-memory",
    default=False,
    help="DataLoader pin_memory. Recommended True on CUDA.",
)
@click.option(
    "--prefetch-factor",
    type=int,
    default=2,
    help="DataLoader prefetch_factor per worker. PyTorch default is 2; "
         "set 4 on A100 with --num-workers 16+ to keep the GPU queue full. "
         "Ignored when --num-workers 0.",
)
def main(
    data_dir: Path,
    architecture: str,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    val_fraction: float,
    seed: int,
    n_samples_per_epoch: int,
    device: str,
    unet_base_channels: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> None:
    """C3.a scaffolding: build train/val datasets + balanced sampler + loaders."""
    # mute per-clip skip warnings from the dataset (we expect some short clips)
    logging.getLogger("fathom.detection.ml_data").setLevel(logging.ERROR)

    target_mode = "heatmap" if architecture == "resnet18" else "mask"
    device_obj = _autodetect_device() if device == "auto" else torch.device(device)

    clip_paths = sorted(data_dir.glob("*.wav"))
    if not clip_paths:
        raise click.UsageError(f"no .wav files under {data_dir}")
    CONSOLE.print(f"[cyan]Found {len(clip_paths)} clips under {data_dir}[/cyan]")

    train_paths, val_paths = _clip_level_train_val_split(clip_paths, val_fraction, seed)
    CONSOLE.print(
        f"[cyan]Clip-level split (seed={seed}, val_fraction={val_fraction}): "
        f"train={len(train_paths)}, val={len(val_paths)}[/cyan]"
    )

    train_augment = PatchAugmentation(
        time_flip_prob=0.5,
        freq_shift_max_bins=2,
        noise_std=0.5,
        target_mode=target_mode,
    )
    train_ds = SyntheticPatchDataset(
        clip_paths=train_paths,
        lofar_config=default_lofar_config(),
        patch_config=PatchExtractionConfig(
            patch_size=256, stride=128, target_mode=target_mode,
        ),
        transform=train_augment,
    )
    val_ds = SyntheticPatchDataset(
        clip_paths=val_paths,
        lofar_config=default_lofar_config(),
        patch_config=PatchExtractionConfig(
            patch_size=256, stride=128, target_mode=target_mode,
        ),
    )
    CONSOLE.print(
        f"[cyan]Train: {len(train_ds)} patches across {len(train_ds._clip_entries)} usable clips[/cyan]"
    )
    CONSOLE.print(
        f"[cyan]Val:   {len(val_ds)} patches across {len(val_ds._clip_entries)} usable clips[/cyan]"
    )

    CONSOLE.print("[cyan]Computing balanced sampler weights from train labels...[/cyan]")
    train_labels = train_ds.get_all_binary_labels()
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    CONSOLE.print(
        f"[cyan]Train labels: positive={n_pos} ({n_pos/len(train_labels)*100:.1f}%), "
        f"negative={n_neg}[/cyan]"
    )

    train_sampler = make_balanced_patch_sampler(
        train_labels, num_samples=n_samples_per_epoch,
    )

    dataloader_kwargs: dict = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler, **dataloader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **dataloader_kwargs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    CONSOLE.print(
        f"\n[cyan]Architecture:[/cyan] {architecture}  "
        f"[cyan]target_mode:[/cyan] {target_mode}  "
        f"[cyan]device:[/cyan] {device_obj}"
    )
    CONSOLE.print(
        f"[cyan]Augmentation:[/cyan] time_flip_p=0.5, freq_shift±2, noise_std=0.5"
    )
    CONSOLE.print(
        f"[cyan]Train:[/cyan] {n_samples_per_epoch} samples/epoch, batch={batch_size}"
    )
    CONSOLE.print(
        f"[cyan]Val:[/cyan] {len(val_ds)} patches, batch={batch_size}"
    )

    # C3.a scaffolding verification: pull a few batches; confirm balance + shapes
    CONSOLE.print("\n[yellow]Sanity check (C3.a scaffolding):[/yellow]")
    sample_train = next(iter(train_loader))
    sample_val = next(iter(val_loader))
    CONSOLE.print(
        f"  Train batch shapes: patch={tuple(sample_train[0].shape)}, "
        f"label={tuple(sample_train[1].shape)}, target={tuple(sample_train[2].shape)}"
    )
    CONSOLE.print(
        f"  Val batch shapes:   patch={tuple(sample_val[0].shape)}, "
        f"label={tuple(sample_val[1].shape)}, target={tuple(sample_val[2].shape)}"
    )

    # Sampler balance check on 20 batches
    pos_count = total_count = 0
    for i, (_patch, binary_labels, _target) in enumerate(train_loader):
        if i >= 20:
            break
        pos_count += int(binary_labels.sum().item())
        total_count += int(binary_labels.numel())
    CONSOLE.print(
        f"  Sampler balance over 20 batches: "
        f"{pos_count}/{total_count} positive "
        f"({pos_count / total_count * 100:.1f}%, expected ~50%)"
    )

       # ---- C3.c: training loop ----
        # ---- C3.c+d: training loop + persistence ----
    CONSOLE.print(f"\n[cyan]Building model + loss + optimizer for {architecture}...[/cyan]")
    torch.manual_seed(seed)
    model = build_model(
        architecture,
        num_freq_bins=256,
        unet_base_channels=unet_base_channels,
    ).to(device_obj)
    loss_fn = build_loss(architecture).to(device_obj)
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    CONSOLE.print(f"[cyan]Model parameters: {n_params:,}[/cyan]")

    run_dir = output_dir / f"{architecture}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = TrainingRunMetadata(
        architecture=architecture,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        n_samples_per_epoch=n_samples_per_epoch,
        val_fraction=val_fraction,
        data_dir=str(data_dir),
        output_dir=str(run_dir),
        device=str(device_obj),
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        augmentation={
            "time_flip_prob": 0.5,
            "freq_shift_max_bins": 2,
            "noise_std": 0.5,
        },
        n_train_clips=len(train_ds._clip_entries),
        n_val_clips=len(val_ds._clip_entries),
        n_train_patches=len(train_ds),
        n_val_patches=len(val_ds),
        started_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    save_run_metadata(metadata, run_dir / "config.json")
    CONSOLE.print(f"[cyan]Run dir: {run_dir}[/cyan]")

    metrics_logger = MetricsLogger(csv_path=run_dir / "metrics.csv")
    best_val = float("inf")

    for epoch in range(epochs):
        # U-Net clDice warmup: advance epoch BEFORE training this epoch
        if architecture == "unet" and hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device_obj, architecture,
        )
        val_metrics = evaluate(
            model, val_loader, loss_fn, device_obj, architecture,
        )
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        CONSOLE.print(
            f"[green]epoch {epoch + 1}/{epochs}[/green]  "
            f"lr={lr:.2e}  "
            f"train_total={train_metrics['total']:.4f}  "
            f"val_total={val_metrics['total']:.4f}  "
            f"train_batches={int(train_metrics['n_batches'])}  "
            f"val_batches={int(val_metrics['n_batches'])}"
        )
        component_keys = [k for k in train_metrics if k not in ("total", "n_batches")]
        if component_keys:
            train_parts = "  ".join(
                f"{k}={train_metrics[k]:.4f}" for k in sorted(component_keys)
            )
            val_parts = "  ".join(
                f"{k}={val_metrics[k]:.4f}" for k in sorted(component_keys)
            )
            CONSOLE.print(f"  [dim]train:[/dim] {train_parts}")
            CONSOLE.print(f"  [dim]val:  [/dim] {val_parts}")

        # Persist this epoch's metrics
        metrics_logger.append(
            epoch=epoch + 1,
            learning_rate=lr,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        # Always save last; conditionally save best
        save_checkpoint(
            run_dir / "last.pt",
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            epoch=epoch + 1,
            architecture=architecture,
            val_metric=val_metrics["total"],
        )
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(
                run_dir / "best.pt",
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                epoch=epoch + 1,
                architecture=architecture,
                val_metric=val_metrics["total"],
            )
            CONSOLE.print(f"  [dim]\u2192 best.pt updated (val_total={best_val:.4f})[/dim]")

    metrics_logger.plot_losses(run_dir / "losses.png")
    CONSOLE.print(f"\n[green]--- C3.d training complete ---[/green]")
    CONSOLE.print(f"  config.json:  {run_dir / 'config.json'}")
    CONSOLE.print(f"  metrics.csv:  {run_dir / 'metrics.csv'}")
    CONSOLE.print(f"  best.pt:      {run_dir / 'best.pt'}  (val_total={best_val:.4f})")
    CONSOLE.print(f"  last.pt:      {run_dir / 'last.pt'}")
    CONSOLE.print(f"  losses.png:   {run_dir / 'losses.png'}")

    # ---- C3.e: Tier-1 evaluation on val set ----
    CONSOLE.print("\n[cyan]Running Tier-1 evaluation on val set...[/cyan]")
    eval_metrics = evaluate_model(model, val_ds, device_obj, architecture)
    print_eval_summary(eval_metrics, console=CONSOLE)

    eval_json_path = run_dir / "tier1_metrics.json"
    import json as _json
    eval_json_path.write_text(_json.dumps(eval_metrics, indent=2))
    CONSOLE.print(f"\n  tier1_metrics.json: {eval_json_path}")


if __name__ == "__main__":
    main()