"""Ambient noise loader.

B1 substitution: DeepShip vessel-free recordings used as ambient stand-in
pending NOAA NRS acquisition (CEO direction 2026-05-10). C1 swaps in NOAA
NRS when downloads are unblocked.

Per A1 §3.1 the ambient layer provides real ocean noise. DeepShip's Halifax-
harbor recordings are real ocean noise; the substitution narrows geographic
coverage but preserves the "real not synthetic" property.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from ..ingestion._resample import resample_to

LOG = logging.getLogger(__name__)


def load_deepship_ambient(path: Path, target_sr: int = 32000) -> tuple[np.ndarray, int]:
    """Load a DeepShip recording, mono-reduce, resample to target_sr.

    Returns (waveform, source_sample_rate).
    """
    wav, source_sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype("float32")
    if int(source_sr) != target_sr:
        LOG.info("resampling ambient %s: %d -> %d Hz", path.name, source_sr, target_sr)
        wav = resample_to(wav, source_sr=int(source_sr), target_sr=target_sr)
    return wav, int(source_sr)