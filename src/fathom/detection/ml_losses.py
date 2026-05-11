"""Loss functions for the ML line detector (A2 §loss).

Patch-CNN dual-head:
  - Binary classification head: focal loss (Lin et al. 2017), γ=2 per A2 baseline
  - Frequency-axis heatmap head: BCE with logits (multi-line capable)
  - Combined: L = L_classification + λ · L_heatmap, λ=1 per A2 default

Why focal loss for classification:
  - Synthetic training data has class imbalance (some patches no tonal coverage,
    some many active bins). γ=2 downweights well-classified examples, focuses
    learning on hard examples (low-SNR / partial-coverage patches).

Why BCE (not single-line regression) for heatmap:
  - Real LOFAR patches commonly carry 3-8 simultaneous tonals (harmonics +
    multi-vessel + machinery). Single-line regression fails on the common case.
  - Per-bin BCE handles arbitrary multi-line activations naturally.

Heatmap loss is computed on ALL patches (positive + negative). Negative
patches have all-zero heatmap_target so the network learns to predict zero
when no line. A2's "activated only on positive patches" refers to inference-
time gating, not training-time masking.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = -1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for binary classification with logits (Lin et al. 2017).

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Args:
        logits: shape (B,) or (B, 1) — pre-sigmoid scores
        targets: shape (B,) — binary targets {0, 1}, cast to float
        gamma: focusing parameter (γ=2 per A2 default; γ=0 collapses to BCE)
        alpha: balancing parameter; -1 disables α-balancing
        reduction: 'mean', 'sum', or 'none'
    """
    if logits.dim() == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # p_t = p if target==1 else 1-p
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)

    focal_modulator = (1.0 - p_t) ** gamma
    loss = focal_modulator * bce

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def heatmap_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """BCE-with-logits for frequency-axis heatmap targets.

    Args:
        logits: shape (B, F) — pre-sigmoid scores per freq bin
        targets: shape (B, F) — binary targets per freq bin (dtype float or bool)
    """
    return F.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction=reduction
    )


class DualHeadLoss(nn.Module):
    """Combined loss for the ResNet-18 patch-CNN dual-head model.

    Forward returns a dict with:
      - total: scalar tensor for .backward()
      - classification: detached scalar for logging
      - heatmap: detached scalar for logging
    """

    def __init__(self, focal_gamma: float = 2.0, heatmap_weight: float = 1.0):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.heatmap_weight = heatmap_weight

    def forward(
        self,
        class_logits: torch.Tensor,
        heatmap_logits: torch.Tensor,
        binary_targets: torch.Tensor,
        heatmap_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l_class = sigmoid_focal_loss(
            class_logits, binary_targets, gamma=self.focal_gamma
        )
        l_heatmap = heatmap_bce_loss(heatmap_logits, heatmap_targets)
        total = l_class + self.heatmap_weight * l_heatmap
        return {
            "total": total,
            "classification": l_class.detach(),
            "heatmap": l_heatmap.detach(),
        }