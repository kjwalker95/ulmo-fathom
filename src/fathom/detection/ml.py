"""ResNet-18 patch-CNN line detector (A2 §architecture).

Primary ML detector for Sprint 4. Consumes 256×256 LOFAR patches (single-
channel, dB scale) and emits two heads:
  - class_logits: (B,) — pre-sigmoid binary "line present in patch?"
  - heatmap_logits: (B, num_freq_bins) — pre-sigmoid per-freq-bin scores

Backbone: torchvision ResNet-18, pretrained on ImageNet by default.
Conv1 channel-averaging surgery converts the 3-channel ImageNet kernel
to a 1-channel kernel for spectrogram input (averages across the channel
dim of pretrained weights; preserves coarse low-level filters).

Heads:
  - class_head: Linear(512, 1) on post-avgpool features
  - heatmap_head: Linear(512, num_freq_bins) on same features

A2 deltas vs naive baseline: surgery on conv1 instead of channel-replication
or scratch init (compared in Sprint 5 ablation). Sigmoid+threshold gating at
inference; heatmap head used only when classifier says positive.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchCNNDetector(nn.Module):
    """ResNet-18 patch-CNN with binary + heatmap heads.

    Args:
        num_freq_bins: heatmap output dim. Match the patch's frequency axis
          size (256 for A2 baseline).
        pretrained: load ImageNet weights and apply channel-averaging surgery.
          False initializes conv1 with default Kaiming init (Sprint 5 ablation).
    """

    def __init__(
        self,
        num_freq_bins: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # Conv1 channel-averaging surgery: (64, 3, 7, 7) → (64, 1, 7, 7).
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            if pretrained:
                new_conv.weight.copy_(
                    old_conv.weight.detach().mean(dim=1, keepdim=True)
                )
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias.detach())
        backbone.conv1 = new_conv

        feature_dim = backbone.fc.in_features  # 512 for ResNet-18
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.class_head = nn.Linear(feature_dim, 1)
        self.heatmap_head = nn.Linear(feature_dim, num_freq_bins)

        self.num_freq_bins = num_freq_bins

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: (B, 1, H, W) — single-channel LOFAR patch
        Returns:
            (class_logits (B,), heatmap_logits (B, num_freq_bins))
        """
        features = self.backbone(x)              # (B, 512)
        class_logits = self.class_head(features).squeeze(-1)  # (B,)
        heatmap_logits = self.heatmap_head(features)          # (B, num_freq_bins)
        return class_logits, heatmap_logits