"""Polyphase resampling for the ingestion layer.

Sprint 3 ShipsEar carries 52,734 Hz native; downstream LOFAR consumes 32 kHz
to match the Sprint 1 default. scipy.signal.resample_poly is the chosen primitive
(polyphase filtering with implicit anti-aliasing). DeepShip is 32 kHz native so
the path is a no-op there.

Per Sprint3_Plan §3, the resampling primitive lives in the ingestion layer and
consumers call it via `resample_to(waveform, source_sr, target_sr)`. This is
platform-layer infrastructure (PCD v3 §6.1 ingestion is platform); Tuor and
future Fathom products consume it without re-deriving the rate-conversion path.
"""
from __future__ import annotations

import logging
from math import gcd

import numpy as np
from scipy.signal import resample_poly

LOG = logging.getLogger(__name__)


def resample_to(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1D mono waveform from source_sr to target_sr via polyphase filtering.

    No-op when source_sr == target_sr. Reduces (up, down) by their GCD before
    calling scipy.signal.resample_poly so e.g. 52,734 -> 32,000 reduces to
    16,000 / 26,367.

    Anti-aliasing is implicit in the polyphase filter chosen by scipy
    (Kaiser-windowed FIR by default). For Tuor / Fathom workloads at 32 kHz
    target rate this fidelity is adequate; higher-fidelity alternatives can
    land later if a downstream consumer measures the resampling artifact as
    material.
    """
    if waveform.ndim != 1:
        raise ValueError(f"expected mono 1D waveform; got shape {waveform.shape}")
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError(
            f"sample rates must be positive; got source={source_sr}, target={target_sr}"
        )
    if source_sr == target_sr:
        return waveform

    common = gcd(int(target_sr), int(source_sr))
    up = int(target_sr) // common
    down = int(source_sr) // common
    LOG.debug("resample_poly: %d -> %d (up=%d, down=%d)", source_sr, target_sr, up, down)
    return resample_poly(waveform, up=up, down=down).astype(waveform.dtype, copy=False)