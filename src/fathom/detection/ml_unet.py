"""U-Net line detector with clDice loss (A2 §architecture parallel).

Parallel architecture to PatchCNNDetector for the Sprint 5 bakeoff. Same
data interface as the patch-CNN (256×256 LOFAR patches), but outputs a dense
2D segmentation mask instead of binary class + 1D heatmap. Loss in
fathom.detection.ml_losses.UNetCombinedLoss (BCE + Dice + clDice with warmup).

Standard Ronneberger U-Net topology:
  - 4 encoder levels: inc (in→c) + 4 × Down (c → 2c each step), bottleneck at c·16
  - 4 decoder levels: each upsamples and concats encoder skip → DoubleConv
  - Output: 1×1 conv to single-channel pre-sigmoid logits

Output shape: (B, H, W) pre-sigmoid logits. Apply torch.sigmoid + threshold
at inference; connected-component analysis on the thresholded mask extracts
LineOfInterest records (C4 onward).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    """Two consecutive Conv2d-BatchNorm-ReLU blocks (U-Net building unit)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    """Encoder block: MaxPool 2× → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    """Decoder block: ConvTranspose 2× upsample → concat skip → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x_below: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x_below)
        # Pad if shape mismatch (input H or W not divisible by 16)
        diff_h = x_skip.size(2) - x.size(2)
        diff_w = x_skip.size(3) - x.size(3)
        if diff_h or diff_w:
            x = F.pad(
                x,
                [
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2,
                ],
            )
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)


class UNetDetector(nn.Module):
    """4-level U-Net for LOFAR line segmentation.

    Args:
        in_channels: input channels (1 for single-channel spectrogram)
        base_channels: encoder's first DoubleConv output channels.
          64 = Ronneberger canonical. Drop to 32 for memory ablation.

    Forward:
        x: (B, in_channels, H, W) — H, W should be divisible by 16 for clean shapes
        Returns: (B, H, W) pre-sigmoid logits (channel dim squeezed)
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.inc = _DoubleConv(in_channels, c)
        self.down1 = _Down(c, c * 2)
        self.down2 = _Down(c * 2, c * 4)
        self.down3 = _Down(c * 4, c * 8)
        self.down4 = _Down(c * 8, c * 16)
        self.up1 = _Up(c * 16, c * 8)
        self.up2 = _Up(c * 8, c * 4)
        self.up3 = _Up(c * 4, c * 2)
        self.up4 = _Up(c * 2, c)
        self.outc = nn.Conv2d(c, 1, kernel_size=1)
        self.base_channels = base_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x).squeeze(1)