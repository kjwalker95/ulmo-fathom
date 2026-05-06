"""DEMON (Detection of Envelope Modulation On Noise) gram generation.

Bandpass → square-law envelope → low-pass → decimate → STFT-of-envelope.
Light-touch in Sprint 1 per Sprint1_Plan §3; full DEMON tuning lands in later
sprints when high-speed cavitation analysis becomes a Phase 1 evaluation question.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, decimate, sosfiltfilt, stft

from ..models import DEMONConfig

LOG = logging.getLogger(__name__)


@dataclass
class DEMONGram:
    frequencies_hz: np.ndarray
    times_s: np.ndarray
    power_db: np.ndarray
    config: DEMONConfig


def compute_demon_gram(waveform: np.ndarray, config: DEMONConfig) -> DEMONGram:
    if waveform.ndim != 1:
        raise ValueError(f"expected mono waveform; got shape {waveform.shape}")
    nyq = config.sample_rate / 2
    bp = butter(
        4,
        [config.band_low_hz / nyq, config.band_high_hz / nyq],
        btype="bandpass",
        output="sos",
    )
    bandpassed = sosfiltfilt(bp, waveform)
    envelope = bandpassed ** 2
    lpf = butter(4, config.envelope_lpf_cutoff_hz / nyq, btype="low", output="sos")
    envelope_lpf = sosfiltfilt(lpf, envelope)
    if config.decimation_factor > 1:
        decimated = decimate(envelope_lpf, config.decimation_factor, ftype="iir")
        envelope_sr = config.sample_rate / config.decimation_factor
    else:
        decimated = envelope_lpf
        envelope_sr = config.sample_rate
    decimated = decimated - decimated.mean()
    win = np.hanning(config.n_fft)
    f, t, z = stft(
        decimated,
        fs=envelope_sr,
        window=win,
        nperseg=config.n_fft,
        noverlap=config.n_fft - config.hop_length,
        nfft=config.n_fft,
        return_onesided=True,
        boundary=None,
        padded=False,
    )
    power_db = 10.0 * np.log10(np.abs(z) ** 2 + 1e-10)
    return DEMONGram(frequencies_hz=f, times_s=t, power_db=power_db, config=config)