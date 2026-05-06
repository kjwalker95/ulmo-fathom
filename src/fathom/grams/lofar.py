"""LOFAR (Low-Frequency Analysis and Recording) gram generation.

Linear-frequency spectrograms via STFT, with split-window normalization to
highlight tonal lines against ambient. Linear frequency, NOT mel-scale: PCD v2
§6.2 commitment, since mel-scale perceptually weights frequencies for human
auditory perception which is the wrong choice for operator-facing analysis of
low-frequency tonal lines.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

from ..models import LOFARConfig
from .normalization import split_window_normalize

LOG = logging.getLogger(__name__)


@dataclass
class LOFARGram:
    """Computed LOFAR gram. Frequencies in Hz, times in seconds, intensity in dB."""

    frequencies_hz: np.ndarray
    times_s: np.ndarray
    power_db: np.ndarray             # log-compressed power, pre-normalization
    normalized_power_db: np.ndarray  # split-window normalized
    config: LOFARConfig


def _resolve_window(name: str, n: int) -> np.ndarray:
    name = name.lower()
    if name in ("hanning", "hann"):
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name == "blackman":
        return np.blackman(n)
    raise ValueError(f"unknown window: {name}")


def compute_lofar_gram(waveform: np.ndarray, config: LOFARConfig) -> LOFARGram:
    """Compute a LOFAR gram from a 1D mono waveform.

    `waveform` must already be at the LOFAR config sample rate. Resampling is the
    ingestion layer's responsibility, not this module's.
    """
    if waveform.ndim != 1:
        raise ValueError(f"expected mono waveform; got shape {waveform.shape}")
    win = _resolve_window(config.stft.window, config.stft.window_length)
    f, t, z = stft(
        waveform,
        fs=config.stft.sample_rate,
        window=win,
        nperseg=config.stft.window_length,
        noverlap=config.stft.window_length - config.stft.hop_length,
        nfft=config.stft.n_fft,
        return_onesided=True,
        boundary=None,
        padded=False,
    )
    power = np.abs(z) ** 2  # energy
    band_mask = (f >= config.freq_min_hz) & (f <= config.freq_max_hz)
    f = f[band_mask]
    power = power[band_mask, :]
    power_db = 10.0 * np.log10(power + config.log_epsilon)
    normalized_power_db = split_window_normalize(
        power_db,
        train_window_bins=config.normalization_train_window_bins,
        central_window_bins=config.normalization_central_window_bins,
        gap_bins=config.normalization_gap_bins,
        axis=0,
    )
    return LOFARGram(
        frequencies_hz=f,
        times_s=t,
        power_db=power_db,
        normalized_power_db=normalized_power_db,
        config=config,
    )