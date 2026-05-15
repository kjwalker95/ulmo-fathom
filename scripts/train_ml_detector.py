"""ML detector training entry point (C3 + Sprint 5 mix-and-train).

C3 (Sprint 4): clip-level train/val split, SyntheticPatchDataset
construction (target_mode per architecture), WeightedRandomSampler for
50/50 batch balance, DataLoader, training loop, Tier-1 evaluation.

Sprint 5 additions (2026-05-15):
  - --real-data-dir + --synthetic-ratio + --val-data-dir flags enabling
    A3 §3.1.1 ratio-sweep training. Mix synthetic + real-ambient
    patches at a specified ratio; evaluate against an external Tier-2
    val dataset. Defaults preserve Sprint 4 single-dataset behavior.
  - make_mixed_balanced_sampler enforces BOTH the synthetic/real domain
    mix AND pos/neg balance within each domain via per-patch weights.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import torch
from rich.console import Console
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

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


def make_mixed_balanced_sampler(
    *,
    synthetic_labels: list[bool],
    real_labels: list[bool],
    synthetic_ratio: float,
    num_samples: int,
) -> WeightedRandomSampler:
    """Sampler with both pos/neg balance AND synthetic/real mix discipline
    (Sprint 5 ratio sweep).

    Per-patch weight enforces two constraints simultaneously:
      - Domain mix: expected `synthetic_ratio` of each batch is synthetic.
      - Pos/neg balance within each domain: 50/50 positive vs negative.

    Concatenation order expected: synthetic patches first, then real
    patches. Caller MUST match this in ConcatDataset([syn, real]).

    Weight per patch:
      synthetic positive: synthetic_ratio * 0.5 / n_synthetic_positive
      synthetic negative: synthetic_ratio * 0.5 / n_synthetic_negative
      real positive:      (1 - synthetic_ratio) * 0.5 / n_real_positive
      real negative:      (1 - synthetic_ratio) * 0.5 / n_real_negative

    WeightedRandomSampler normalizes weights internally so only ratios
    matter. Patches in a zero-count bucket get weight 0 (skipped). Mix
    ratios of 0.0 or 1.0 cause one domain to be entirely skipped.
    """
    if not (0.0 <= synthetic_ratio <= 1.0):
        raise ValueError(
            f"synthetic_ratio must be in [0, 1]; got {synthetic_ratio}"
        )
    if not synthetic_labels and not real_labels:
        raise ValueError("both synthetic_labels and real_labels are empty")

    n_syn_pos = sum(synthetic_labels)
    n_syn_neg = len(synthetic_labels) - n_syn_pos
    n_real_pos = sum(real_labels)
    n_real_neg = len(real_labels) - n_real_pos

    def _w(domain_mix: float, label: bool, n_pos: int, n_neg: int) -> float:
        if domain_mix <= 0.0:
            return 0.0
        if label:
            return (domain_mix * 0.5 / n_pos) if n_pos > 0 else 0.0
        return (domain_mix * 0.5 / n_neg) if n_neg > 0 else 0.0

    weights: list[float] = []
    for label in synthetic_labels:
        weights.append(_w(synthetic_ratio, label, n_syn_pos, n_syn_neg))
    for label in real_labels:
        weights.append(_w(1.0 - synthetic_ratio, label, n_real_pos, n_real_neg))

    if sum(weights) <= 0.0:
        raise ValueError(
            f"all sampler weights are zero — check ratio + dataset sizes: "
            f"synthetic_ratio={synthetic_ratio} "
            f"n_synthetic={len(synthetic_labels)} n_real={len(real_labels)}"
        )

    return WeightedRandomSampler(
        weights=weights, num_samples=num_samples, replacement=True,
    )


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Synthetic training dataset directory (output of build_training_dataset.py). "
         "Sprint 4 single-dataset behavior preserved when --real-data-dir is omitted.",
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
@click.option(
    "--real-data-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional Tier-2-style real-ambient training dataset (Sprint 5 §C2 "
         "ratio sweep: data/tier2_train_v2). When provided, training mixes "
         "patches from --data-dir (synthetic) and this dataset per "
         "--synthetic-ratio. Omit to preserve Sprint 4 single-dataset behavior.",
)
@click.option(
    "--synthetic-ratio",
    type=float,
    default=1.0,
    help="Synthetic fraction in each batch when --real-data-dir is provided. "
         "1.0 = all synthetic (Sprint 4 default); 0.0 = all real. "
         "Sprint 5 ratio sweep cells: 0/0.25/0.38/0.50/0.75/0.90/1.0.",
)
@click.option(
    "--val-data-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional external validation dataset (Sprint 5: data/tier2_val_v2). "
         "When provided, val patches come from this dataset and --val-fraction "
         "is ignored. Required for the Sprint 5 ratio sweep to evaluate "
         "against Tier-2 val. Omit to preserve Sprint 4 clip-level val split.",
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
    real_data_dir: Path | None,
    synthetic_ratio: float,
    val_data_dir: Path | None,
) -> None:
    """ML detector training (Sprint 4 single-dataset + Sprint 5 mix-and-train)."""
    logging.getLogger("fathom.detection.ml_data").setLevel(logging.ERROR)

    target_mode = "heatmap" if architecture == "resnet18" else "mask"
    device_obj = _autodetect_device() if device == "auto" else torch.device(device)

    # ---- enumerate synthetic + (optional) real training clips ----
    synthetic_clip_paths = sorted(data_dir.glob("*.wav"))
    if not synthetic_clip_paths:
        raise click.UsageError(f"no .wav files under {data_dir}")
    CONSOLE.print(
        f"[cyan]Synthetic clips: {len(synthetic_clip_paths)} under {data_dir}[/cyan]"
    )

    real_clip_paths: list[Path] = []
    if real_data_dir is not None:
        real_clip_paths = sorted(real_data_dir.glob("*.wav"))
        if not real_clip_paths:
            raise click.UsageError(f"no .wav files under {real_data_dir}")
        CONSOLE.print(
            f"[cyan]Real clips: {len(real_clip_paths)} under {real_data_dir}[/cyan]"
        )
        if not (0.0 <= synthetic_ratio <= 1.0):
            raise click.UsageError(
                f"--synthetic-ratio must be in [0, 1]; got {synthetic_ratio}"
            )
    else:
        if synthetic_ratio != 1.0:
            CONSOLE.print(
                "[yellow]warn: --synthetic-ratio != 1.0 but no --real-data-dir; "
                "ratio will be ignored (Sprint 4 single-dataset behavior)[/yellow]"
            )

    # ---- resolve train and val partitions ----
    if val_data_dir is not None:
        synthetic_train_paths = synthetic_clip_paths
        real_train_paths = real_clip_paths
        val_paths = sorted(val_data_dir.glob("*.wav"))
        if not val_paths:
            raise click.UsageError(f"no .wav files under {val_data_dir}")
        CONSOLE.print(
            f"[cyan]External val: {len(val_paths)} clips under {val_data_dir}[/cyan]"
        )
    else:
        synthetic_train_paths, val_paths = _clip_level_train_val_split(
            synthetic_clip_paths, val_fraction, seed,
        )
        real_train_paths = real_clip_paths
        CONSOLE.print(
            f"[cyan]Clip-level synthetic split (seed={seed}, "
            f"val_fraction={val_fraction}): "
            f"train={len(synthetic_train_paths)}, val={len(val_paths)}[/cyan]"
        )

    # ---- build datasets ----
    train_augment = PatchAugmentation(
        time_flip_prob=0.5,
        freq_shift_max_bins=2,
        noise_std=0.5,
        target_mode=target_mode,
    )
    patch_config = PatchExtractionConfig(
        patch_size=256, stride=128, target_mode=target_mode,
    )
    synthetic_train_ds = SyntheticPatchDataset(
        clip_paths=synthetic_train_paths,
        lofar_config=default_lofar_config(),
        patch_config=patch_config,
        transform=train_augment,
    )
    CONSOLE.print(
        f"[cyan]Synthetic train: {len(synthetic_train_ds)} patches "
        f"across {len(synthetic_train_ds._clip_entries)} usable clips[/cyan]"
    )

    real_train_ds: SyntheticPatchDataset | None = None
    if real_train_paths:
        real_train_ds = SyntheticPatchDataset(
            clip_paths=real_train_paths,
            lofar_config=default_lofar_config(),
            patch_config=patch_config,
            transform=train_augment,
        )
        CONSOLE.print(
            f"[cyan]Real train: {len(real_train_ds)} patches "
            f"across {len(real_train_ds._clip_entries)} usable clips[/cyan]"
        )

    val_ds = SyntheticPatchDataset(
        clip_paths=val_paths,
        lofar_config=default_lofar_config(),
        patch_config=patch_config,
    )
    CONSOLE.print(
        f"[cyan]Val: {len(val_ds)} patches across "
        f"{len(val_ds._clip_entries)} usable clips[/cyan]"
    )

    # ---- sampler: balanced + (optionally) mixed across domains ----
    CONSOLE.print("[cyan]Computing sampler weights from train labels...[/cyan]")
    synthetic_labels = synthetic_train_ds.get_all_binary_labels()
    if real_train_ds is not None:
        real_labels = real_train_ds.get_all_binary_labels()
        train_sampler = make_mixed_balanced_sampler(
            synthetic_labels=synthetic_labels,
            real_labels=real_labels,
            synthetic_ratio=synthetic_ratio,
            num_samples=n_samples_per_epoch,
        )
        train_ds = ConcatDataset([synthetic_train_ds, real_train_ds])
        n_pos_syn = sum(synthetic_labels)
        n_neg_syn = len(synthetic_labels) - n_pos_syn
        n_pos_real = sum(real_labels)
        n_neg_real = len(real_labels) - n_pos_real
        CONSOLE.print(
            f"[cyan]Synthetic labels: positive={n_pos_syn} "
            f"({n_pos_syn / max(1, len(synthetic_labels)) * 100:.1f}%), "
            f"negative={n_neg_syn}[/cyan]"
        )
        CONSOLE.print(
            f"[cyan]Real labels:      positive={n_pos_real} "
            f"({n_pos_real / max(1, len(real_labels)) * 100:.1f}%), "
            f"negative={n_neg_real}[/cyan]"
        )
        CONSOLE.print(
            f"[cyan]Mix: synthetic_ratio={synthetic_ratio}; "
            f"expected per-batch ~{int(batch_size * synthetic_ratio)} synthetic / "
            f"~{int(batch_size * (1 - synthetic_ratio))} real[/cyan]"
        )
    else:
        train_sampler = make_balanced_patch_sampler(
            synthetic_labels, num_samples=n_samples_per_epoch,
        )
        train_ds = synthetic_train_ds
        n_pos = sum(synthetic_labels)
        n_neg = len(synthetic_labels) - n_pos
        CONSOLE.print(
            f"[cyan]Train labels: positive={n_pos} "
            f"({n_pos / max(1, len(synthetic_labels)) * 100:.1f}%), negative={n_neg}[/cyan]"
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

    # Scaffolding sanity check
    CONSOLE.print("\n[yellow]Sanity check (scaffolding):[/yellow]")
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

    pos_count = total_count = 0
    for i, (_patch, binary_labels, _target) in enumerate(train_loader):
        if i >= 20:
            break
        pos_count += int(binary_labels.sum().item())
        total_count += int(binary_labels.numel())
    CONSOLE.print(
        f"  Sampler balance over 20 batches: "
        f"{pos_count}/{total_count} positive "
        f"({pos_count / max(1, total_count) * 100:.1f}%, expected ~50%)"
    )

    # ---- training loop ----
    CONSOLE.print(
        f"\n[cyan]Building model + loss + optimizer for {architecture}...[/cyan]"
    )
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

    if real_train_ds is not None:
        run_dir_name = (
            f"{architecture}_seed{seed}_ratio{synthetic_ratio:.2f}"
        )
    else:
        run_dir_name = f"{architecture}_seed{seed}"
    run_dir = output_dir / run_dir_name
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
            "real_data_dir": str(real_data_dir) if real_data_dir else None,
            "synthetic_ratio": synthetic_ratio,
            "val_data_dir": str(val_data_dir) if val_data_dir else None,
        },
        n_train_clips=(
            len(synthetic_train_ds._clip_entries)
            + (len(real_train_ds._clip_entries) if real_train_ds else 0)
        ),
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

        metrics_logger.append(
            epoch=epoch + 1,
            learning_rate=lr,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

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
            CONSOLE.print(
                f"  [dim]\u2192 best.pt updated (val_total={best_val:.4f})[/dim]"
            )

    metrics_logger.plot_losses(run_dir / "losses.png")
    CONSOLE.print(f"\n[green]--- training complete ---[/green]")
    CONSOLE.print(f"  config.json:  {run_dir / 'config.json'}")
    CONSOLE.print(f"  metrics.csv:  {run_dir / 'metrics.csv'}")
    CONSOLE.print(f"  best.pt:      {run_dir / 'best.pt'}  (val_total={best_val:.4f})")
    CONSOLE.print(f"  last.pt:      {run_dir / 'last.pt'}")
    CONSOLE.print(f"  losses.png:   {run_dir / 'losses.png'}")

    # ---- patch-level evaluation on val set ----
    # When --val-data-dir is given, this is Tier-2 eval (Sprint 5 ratio sweep);
    # otherwise it's Tier-1 eval on the clip-level val split (Sprint 4 behavior).
    CONSOLE.print("\n[cyan]Running patch-level evaluation on val set...[/cyan]")
    eval_metrics = evaluate_model(model, val_ds, device_obj, architecture)
    print_eval_summary(eval_metrics, console=CONSOLE)

    eval_json_path = run_dir / "tier1_metrics.json"
    import json as _json
    eval_json_path.write_text(_json.dumps(eval_metrics, indent=2))
    CONSOLE.print(f"\n  tier1_metrics.json: {eval_json_path}")


if __name__ == "__main__":
    main()