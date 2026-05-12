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
    *,
    pos_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """BCE-with-logits for frequency-axis heatmap targets.

    Args:
        logits: shape (B, F) — pre-sigmoid scores per freq bin
        targets: shape (B, F) — binary targets per freq bin (float or bool)
        pos_weight: optional scalar or (F,) tensor — multiplies the positive-
            class term in BCE. Use to counter the severe class imbalance on
            sparse heatmap targets (~5 positive bins per 256 total in our case).
    """
    return F.binary_cross_entropy_with_logits(
        logits, targets.float(),
        pos_weight=pos_weight,
        reduction=reduction,
    )


class DualHeadLoss(nn.Module):
    """Combined loss for the ResNet-18 patch-CNN dual-head model.

    Combines focal loss on the binary classification head (A2 §loss) with
    pos-weighted BCE on the frequency-axis heatmap head. The pos_weight
    counters the severe class imbalance: typical positive patches have
    ~3-10 active bins of 256, so unweighted BCE collapses to "predict zero
    everywhere" as the loss-minimizing strategy. heatmap_pos_weight=50 is
    roughly the inverse of the positive-bin density and brings the head
    out of the all-zeros local minimum.

    Forward returns a dict with:
      - total: scalar tensor for .backward()
      - classification: detached scalar for logging
      - heatmap: detached scalar for logging
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        heatmap_weight: float = 1.0,
        heatmap_pos_weight: float = 50.0,
    ):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.heatmap_weight = heatmap_weight
        self.heatmap_pos_weight = float(heatmap_pos_weight)

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
        pos_weight = torch.tensor(
            self.heatmap_pos_weight,
            dtype=heatmap_logits.dtype,
            device=heatmap_logits.device,
        )
        l_heatmap = heatmap_bce_loss(
            heatmap_logits, heatmap_targets, pos_weight=pos_weight,
        )
        total = l_class + self.heatmap_weight * l_heatmap
        return {
            "total": total,
            "classification": l_class.detach(),
            "heatmap": l_heatmap.detach(),
        }

    

# ===========================================================================
# C2.2: U-Net segmentation losses (Dice + clDice + combined, A2 §loss parallel)
# ===========================================================================


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Soft erosion via -max_pool(-img). Asymmetric (3, 1) kernel thins along
    the freq axis (axis -2) while preserving the time axis (axis -1)."""
    return -F.max_pool2d(-img, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize_2d(img: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
    """Differentiable approximation of skeletonization (Shit et al. CVPR 2021).

    Iterative morphological thinning: each iteration erodes the image,
    extracts the residual (difference between input and opened input), and
    accumulates that residual into the running skeleton via a soft OR.

    Input: (B, H, W) or (B, 1, H, W) probability map in [0, 1].
    Output: same shape, soft skeleton.
    Asymmetric (3, 1) kernel: thins along freq axis (H), preserves time axis (W).
    """
    added_channel = img.dim() == 3
    if added_channel:
        img = img.unsqueeze(1)  # (B, 1, H, W)

    img1 = _soft_open(img)
    skel = F.relu(img - img1)

    for _ in range(n_iter):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        # Soft OR: skel = skel + delta * (1 - skel)
        skel = skel + F.relu(delta - skel * delta)

    return skel.squeeze(1) if added_channel else skel


def dice_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Standard Dice loss (Sørensen–Dice).

    Dice = 2|P ∩ T| / (|P| + |T|);  loss = 1 - Dice.
    Inputs must be in [0, 1] (apply sigmoid to logits before calling).
    """
    preds = preds.float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    return 1.0 - (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def cldice_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    n_iter: int = 10,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Centerline Dice loss (Shit et al. CVPR 2021).

    Encourages topological connectivity for thin tubular foreground (vessels
    in the paper; tonal lines for us).

    T_prec = (S_P · V_L).sum() / S_P.sum()        — topology precision
    T_sens = (S_L · V_P).sum() / S_L.sum()        — topology sensitivity
    clDice = 2 · T_prec · T_sens / (T_prec + T_sens)
    loss = 1 - clDice

    Inputs must be in [0, 1] (apply sigmoid to logits before calling).
    """
    preds = preds.float()
    targets = targets.float()

    skel_p = soft_skeletonize_2d(preds, n_iter)
    skel_l = soft_skeletonize_2d(targets, n_iter)

    t_prec = (skel_p * targets).sum() / (skel_p.sum() + smooth)
    t_sens = (skel_l * preds).sum() / (skel_l.sum() + smooth)
    cldice = 2.0 * t_prec * t_sens / (t_prec + t_sens + 1e-6)
    return 1.0 - cldice


class UNetCombinedLoss(nn.Module):
    """A2 §loss parallel: L = L_BCE + α·L_Dice + β·L_clDice.

    Defaults α=1, β=0.5 per clDice paper. Warmup: clDice term contributes 0
    before `cldice_warmup_epochs`, then switches on. Call `set_epoch(n)`
    each epoch to advance the warmup state.

    Forward returns dict with components ('total', 'bce', 'dice', 'cldice').
    Apply sigmoid internally so callers pass raw logits.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        cldice_weight: float = 0.5,
        cldice_warmup_epochs: int = 5,
        cldice_n_iter: int = 10,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.cldice_warmup_epochs = cldice_warmup_epochs
        self.cldice_n_iter = cldice_n_iter
        self.dice_smooth = dice_smooth
        self._current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)

    def forward(
        self,
        mask_logits: torch.Tensor,
        mask_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l_bce = F.binary_cross_entropy_with_logits(mask_logits, mask_targets.float())
        preds = torch.sigmoid(mask_logits)
        l_dice = dice_loss(preds, mask_targets, smooth=self.dice_smooth)

        if self._current_epoch >= self.cldice_warmup_epochs:
            l_cldice = cldice_loss(preds, mask_targets, n_iter=self.cldice_n_iter)
        else:
            l_cldice = torch.zeros((), device=mask_logits.device, dtype=mask_logits.dtype)

        total = l_bce + self.dice_weight * l_dice + self.cldice_weight * l_cldice
        return {
            "total": total,
            "bce": l_bce.detach(),
            "dice": l_dice.detach(),
            "cldice": l_cldice.detach(),
        }