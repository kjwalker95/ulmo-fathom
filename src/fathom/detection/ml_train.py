"""ML detector training + evaluation loops (A2 §training-schedule).

Architecture dispatch:
  - resnet18: PatchCNNDetector + DualHeadLoss (focal + heatmap BCE)
  - unet:     UNetDetector + UNetCombinedLoss (BCE + Dice + clDice with warmup)

Both consume the same DataLoader output: (patch, binary_label, target).
Target shape varies by architecture per the dataset's target_mode:
  - resnet18: (B, num_freq_bins) heatmap
  - unet:     (B, H, W) segmentation mask

Optimizer: AdamW (lr=1e-3, weight_decay=1e-4 per A2 baseline).
Scheduler: CosineAnnealingLR over `epochs`.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fathom.detection.ml import PatchCNNDetector
from fathom.detection.ml_losses import DualHeadLoss, UNetCombinedLoss
from fathom.detection.ml_unet import UNetDetector

logger = logging.getLogger(__name__)


def build_model(
    architecture: str,
    num_freq_bins: int = 256,
    *,
    unet_base_channels: int = 64,
) -> nn.Module:
    if architecture == "resnet18":
        return PatchCNNDetector(num_freq_bins=num_freq_bins, pretrained=True)
    if architecture == "unet":
        return UNetDetector(in_channels=1, base_channels=unet_base_channels)
    raise ValueError(f"unknown architecture {architecture!r}; expected 'resnet18' or 'unet'")


def build_loss(
    architecture: str,
    *,
    focal_gamma: float = 2.0,
    heatmap_weight: float = 1.0,
    dice_weight: float = 1.0,
    cldice_weight: float = 0.5,
    cldice_warmup_epochs: int = 5,
) -> nn.Module:
    if architecture == "resnet18":
        return DualHeadLoss(focal_gamma=focal_gamma, heatmap_weight=heatmap_weight)
    if architecture == "unet":
        return UNetCombinedLoss(
            dice_weight=dice_weight,
            cldice_weight=cldice_weight,
            cldice_warmup_epochs=cldice_warmup_epochs,
        )
    raise ValueError(f"unknown architecture {architecture!r}")


def _step(
    model: nn.Module,
    loss_fn: nn.Module,
    architecture: str,
    patch: torch.Tensor,
    binary_label: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """One forward + loss-compute step. Returns loss component dict."""
    if architecture == "resnet18":
        class_logits, heatmap_logits = model(patch)
        return loss_fn(class_logits, heatmap_logits, binary_label, target)
    if architecture == "unet":
        mask_logits = model(patch)
        return loss_fn(mask_logits, target)
    raise ValueError(f"unknown architecture {architecture!r}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    architecture: str,
) -> dict[str, float]:
    """One training epoch. Returns averaged loss-component dict + n_batches."""
    model.train()
    sums: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        patch, binary_label, target = batch
        patch = patch.to(device)
        binary_label = binary_label.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss_dict = _step(model, loss_fn, architecture, patch, binary_label, target)
        loss_dict["total"].backward()
        optimizer.step()

        for k, v in loss_dict.items():
            v_float = v.item() if hasattr(v, "item") else float(v)
            sums[k] = sums.get(k, 0.0) + v_float
        n_batches += 1

    avg = {k: v / n_batches for k, v in sums.items()}
    avg["n_batches"] = n_batches
    return avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    architecture: str,
) -> dict[str, float]:
    """One validation pass. Returns averaged loss-component dict + n_batches."""
    model.eval()
    sums: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        patch, binary_label, target = batch
        patch = patch.to(device)
        binary_label = binary_label.to(device)
        target = target.to(device)

        loss_dict = _step(model, loss_fn, architecture, patch, binary_label, target)
        for k, v in loss_dict.items():
            v_float = v.item() if hasattr(v, "item") else float(v)
            sums[k] = sums.get(k, 0.0) + v_float
        n_batches += 1

    if n_batches == 0:
        return {"n_batches": 0}
    avg = {k: v / n_batches for k, v in sums.items()}
    avg["n_batches"] = n_batches
    return avg