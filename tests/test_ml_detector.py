"""Tests for the ResNet-18 patch-CNN line detector (A2 §architecture).

Covers:
  - sigmoid_focal_loss: γ=0 equivalence to BCE; γ=2 downweighting confident-correct
  - DualHeadLoss: end-to-end gradient flow
  - SyntheticPatchDataset: tensor shapes, negative-clip yields zero labels,
    positive-clip yields some positive labels with active heatmap bins
  - PatchCNNDetector: conv1 surgery preserves channel-avg, forward shapes,
    end-to-end backward through the loss, optimizer step changes weights
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fathom.detection.ml import PatchCNNDetector
from fathom.detection.ml_data import (
    PatchExtractionConfig,
    SyntheticPatchDataset,
    default_lofar_config,
)
from fathom.detection.ml_losses import (
    DualHeadLoss,
    sigmoid_focal_loss,
)
from fathom.synthetic import TonalParameterPriors, generate_c1_1_clip


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_clip(tmp_root, *, seed: int, priors: TonalParameterPriors):
    """Generate a 35s synthetic clip with a controlled prior."""
    sr = 32000
    duration_s = 35.0
    ambient_audio = np.random.default_rng(0).normal(
        0, 0.01, int(sr * duration_s)
    ).astype(np.float32)
    ambient_path = tmp_root / "ambient.wav"
    sf.write(str(ambient_path), ambient_audio, samplerate=sr, subtype="PCM_16")

    out_path = tmp_root / "clip.wav"
    generate_c1_1_clip(
        ambient_path=ambient_path,
        out_path=out_path,
        seed=seed,
        priors=priors,
    )
    return out_path


@pytest.fixture(scope="module")
def positive_clip_path(tmp_path_factory):
    """35s clip with 2 forced tonal sources — guarantees positive patches."""
    return _make_clip(
        tmp_path_factory.mktemp("ml_positive"),
        seed=42,
        priors=TonalParameterPriors(n_sources_distribution={2: 1.0}),
    )


@pytest.fixture(scope="module")
def negative_clip_path(tmp_path_factory):
    """35s clip with 0 forced sources — all patches negative, all heatmaps zero."""
    return _make_clip(
        tmp_path_factory.mktemp("ml_negative"),
        seed=1,
        priors=TonalParameterPriors(n_sources_distribution={0: 1.0}),
    )


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------


def test_focal_loss_equals_bce_at_gamma_zero():
    """sigmoid_focal_loss with γ=0 collapses to standard BCE."""
    torch.manual_seed(0)
    logits = torch.randn(16)
    targets = torch.randint(0, 2, (16,)).float()
    focal = sigmoid_focal_loss(logits, targets, gamma=0.0)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    assert torch.allclose(focal, bce, atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    """γ=2 yields ≥10x smaller loss than γ=0 for confident-correct predictions."""
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    confident_logits = (2 * targets - 1) * 4.0
    fl_g0 = sigmoid_focal_loss(confident_logits, targets, gamma=0.0)
    fl_g2 = sigmoid_focal_loss(confident_logits, targets, gamma=2.0)
    assert fl_g2 < fl_g0 / 10.0


def test_dual_head_loss_gradient_flow():
    """DualHeadLoss backward populates grads on both class + heatmap logits."""
    torch.manual_seed(0)
    B, F_bins = 8, 256
    class_logits = torch.randn(B, requires_grad=True)
    heatmap_logits = torch.randn(B, F_bins, requires_grad=True)
    binary_targets = torch.randint(0, 2, (B,)).float()
    heatmap_targets = (torch.rand(B, F_bins) < 0.05).float()

    loss_fn = DualHeadLoss(focal_gamma=2.0, heatmap_weight=1.0)
    out = loss_fn(class_logits, heatmap_logits, binary_targets, heatmap_targets)
    assert out["total"].item() > 0
    out["total"].backward()
    assert class_logits.grad is not None and class_logits.grad.abs().sum() > 0
    assert heatmap_logits.grad is not None and heatmap_logits.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


def test_patch_dataset_shapes(positive_clip_path):
    """Patches are (1, 256, 256); labels are scalar float; heatmaps are (256,)."""
    ds = SyntheticPatchDataset(
        clip_paths=[positive_clip_path],
        lofar_config=default_lofar_config(),
        patch_config=PatchExtractionConfig(patch_size=256, stride=128),
    )
    assert len(ds) > 0
    patch, label, heatmap = ds[0]
    assert patch.shape == (1, 256, 256)
    assert patch.dtype == torch.float32
    assert label.shape == ()
    assert label.dtype == torch.float32
    assert heatmap.shape == (256,)


def test_patch_dataset_negative_clip_has_zero_labels(negative_clip_path):
    """Every patch in a negative clip yields binary_label=0 and heatmap=zeros."""
    ds = SyntheticPatchDataset(
        clip_paths=[negative_clip_path],
        lofar_config=default_lofar_config(),
        patch_config=PatchExtractionConfig(patch_size=256, stride=128),
    )
    for i in range(len(ds)):
        _, label, heatmap = ds[i]
        assert label.item() == 0.0
        assert heatmap.sum().item() == 0.0


def test_patch_dataset_positive_clip_has_positive_patches(positive_clip_path):
    """A clip with forced tonals yields at least one positive patch with heatmap activations."""
    ds = SyntheticPatchDataset(
        clip_paths=[positive_clip_path],
        lofar_config=default_lofar_config(),
        patch_config=PatchExtractionConfig(patch_size=256, stride=128),
    )
    n_positive = sum(int(ds[i][1].item()) for i in range(len(ds)))
    total_activations = sum(int(ds[i][2].sum().item()) for i in range(len(ds)))
    assert n_positive > 0, "expected at least one positive patch given 2 forced tonals"
    assert total_activations > 0, "expected at least one heatmap bin to be active"


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_resnet18_conv1_surgery_singlechannel():
    """Pretrained conv1 weights are channel-averaged from (64,3,7,7) to (64,1,7,7)."""
    from torchvision.models import ResNet18_Weights, resnet18

    reference = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    expected = reference.conv1.weight.detach().mean(dim=1, keepdim=True)
    model = PatchCNNDetector(num_freq_bins=256, pretrained=True)
    assert model.backbone.conv1.weight.shape == (64, 1, 7, 7)
    assert torch.allclose(model.backbone.conv1.weight, expected, atol=1e-6)


def test_resnet18_forward_pass_shapes():
    """Forward on (4, 1, 256, 256) returns class_logits (4,) + heatmap (4, 256)."""
    torch.manual_seed(0)
    model = PatchCNNDetector(num_freq_bins=256, pretrained=False)
    x = torch.randn(4, 1, 256, 256)
    with torch.no_grad():
        cl, hl = model(x)
    assert cl.shape == (4,)
    assert hl.shape == (4, 256)
    assert not torch.isnan(cl).any()
    assert not torch.isnan(hl).any()


def test_resnet18_gradient_flow_end_to_end(positive_clip_path):
    """Loss.backward populates grads on conv1, class_head, heatmap_head."""
    ds = SyntheticPatchDataset(
        clip_paths=[positive_clip_path],
        lofar_config=default_lofar_config(),
        patch_config=PatchExtractionConfig(patch_size=256, stride=128),
    )
    loader = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=False, num_workers=0)
    patch_batch, label_batch, heatmap_batch = next(iter(loader))

    model = PatchCNNDetector(num_freq_bins=256, pretrained=False)
    loss_fn = DualHeadLoss(focal_gamma=2.0, heatmap_weight=1.0)

    model.train()
    cl, hl = model(patch_batch)
    loss_fn(cl, hl, label_batch, heatmap_batch)["total"].backward()

    for name, p in [
        ("backbone.conv1", model.backbone.conv1.weight),
        ("class_head", model.class_head.weight),
        ("heatmap_head", model.heatmap_head.weight),
    ]:
        assert p.grad is not None, f"{name} has no grad"
        assert p.grad.abs().sum().item() > 0, f"{name} grad is all zero"


def test_resnet18_optimizer_step_changes_weights():
    """An AdamW step actually moves the head weights."""
    torch.manual_seed(0)
    model = PatchCNNDetector(num_freq_bins=256, pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(4, 1, 256, 256)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    heatmap_targets = (torch.rand(4, 256) < 0.05).float()

    loss_fn = DualHeadLoss()
    prev = model.class_head.weight.detach().clone()
    cl, hl = model(x)
    loss_fn(cl, hl, targets, heatmap_targets)["total"].backward()
    opt.step()
    delta = (model.class_head.weight - prev).abs().sum().item()
    assert delta > 0