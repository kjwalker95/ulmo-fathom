"""Patch augmentation for ML detector training (A2 §augmentation).

Three transforms applied to (patch, target) tuples during training:
  - Random time-flip (p=0.5): legitimate for line detection since
    direction-of-time is invariant for tonal signatures
  - Random ±N bin frequency shift (default ±2): mimics small drift at
    sub-second scale; zero-pads the wrap-around region
  - Additive Gaussian on the patch (small σ): simulates SNR variability

A2 explicitly excludes time/frequency masking — would obscure the lines
we're training the detector to find.

Applied to TRAINING only. Validation uses raw patches for clean evaluation.

The transform handles both target shapes:
  - "heatmap" (256,): 1D freq-axis. Time-flip is no-op; freq-shift applies.
  - "mask" (256, 256): 2D segmentation. Both flips and shifts apply.

RNG via torch.rand / torch.randn / torch.randint — uses the global torch
RNG. Seed via torch.manual_seed at training start for reproducibility.
"""
from __future__ import annotations

import torch


class PatchAugmentation:
    """Stateless callable applied to (patch, target) tuples."""

    def __init__(
        self,
        time_flip_prob: float = 0.5,
        freq_shift_max_bins: int = 2,
        noise_std: float = 0.5,
        target_mode: str = "heatmap",
    ):
        if target_mode not in ("heatmap", "mask"):
            raise ValueError(
                f"target_mode must be 'heatmap' or 'mask'; got {target_mode!r}"
            )
        if not (0.0 <= time_flip_prob <= 1.0):
            raise ValueError(f"time_flip_prob must be in [0, 1]; got {time_flip_prob}")
        if freq_shift_max_bins < 0:
            raise ValueError(
                f"freq_shift_max_bins must be >= 0; got {freq_shift_max_bins}"
            )
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0; got {noise_std}")
        self.time_flip_prob = time_flip_prob
        self.freq_shift_max_bins = freq_shift_max_bins
        self.noise_std = noise_std
        self.target_mode = target_mode

    def __call__(
        self,
        patch: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation.

        Args:
            patch: (1, H, W) — single-channel LOFAR patch
            target: (H,) for heatmap mode, (H, W) for mask mode

        Returns:
            (patch_aug, target_aug) — same shapes as inputs.
        """
        # --- Time flip (last axis) ---
        if self.time_flip_prob > 0 and torch.rand(1).item() < self.time_flip_prob:
            patch = patch.flip(-1)
            if self.target_mode == "mask":
                target = target.flip(-1)
            # heatmap is freq-only, time-invariant — no change

        # --- Freq shift ---
        if self.freq_shift_max_bins > 0:
            shift = int(
                torch.randint(
                    -self.freq_shift_max_bins,
                    self.freq_shift_max_bins + 1,
                    (1,),
                ).item()
            )
            if shift != 0:
                patch = self._shift_with_zero_pad(patch, shift, axis=-2)
                # Target axis: freq is dim 0 for both heatmap (H,) and mask (H, W)
                target = self._shift_with_zero_pad(target, shift, axis=0)

        # --- Additive Gaussian (input only) ---
        if self.noise_std > 0:
            patch = patch + torch.randn_like(patch) * self.noise_std

        return patch, target

    @staticmethod
    def _shift_with_zero_pad(
        t: torch.Tensor, shift: int, axis: int,
    ) -> torch.Tensor:
        """Roll along `axis`; zero-pad the wrap region so we don't introduce
        spurious activations from the opposite end of the freq axis."""
        rolled = torch.roll(t, shifts=shift, dims=axis)
        slicer: list[slice] = [slice(None)] * rolled.dim()
        if shift > 0:
            slicer[axis] = slice(0, shift)
        else:
            slicer[axis] = slice(shift, None)
        rolled[tuple(slicer)] = 0.0
        return rolled