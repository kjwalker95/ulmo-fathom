"""Tier-2 real-ambient injection evaluation (A3 §3.1 Tier-2).

Sprint 5 Cluster C1. Primary real-evaluation method: inject known synthetic
tonals into held-out real DeepShip ambient recordings, then evaluate trained
ML detectors against the exact-truth label set. SyntheticPatchDataset and
ml_eval.evaluate_model run unmodified on the produced clips because the
output triplet matches the C1.1 synthetic-clip schema.
"""
from fathom.evaluation.injection import (
    Tier2InjectionResult,
    inject_into_real_ambient,
)
from fathom.evaluation.tier2 import (
    Tier2DatasetManifest,
    build_tier2_dataset,
)

__all__ = [
    "Tier2InjectionResult",
    "inject_into_real_ambient",
    "Tier2DatasetManifest",
    "build_tier2_dataset",
]