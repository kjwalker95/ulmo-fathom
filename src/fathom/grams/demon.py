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
    power = np.abs(z) ** 2
    # Peak-relative dB scale. Real DEMON envelopes have small absolute power
    # (~1e-12 to 1e-9 typical) — a fixed epsilon of 1e-10 squashes everything
    # to the log floor and the gram becomes uniform. Referencing the envelope
    # STFT peak puts the strongest modulation at 0 dB and shows the dynamic
    # range of the modulation content. Operators read DEMON in relative dB
    # anyway; absolute envelope dB is not a useful operational unit.
    ref = max(float(power.max()), float(np.finfo(np.float32).tiny))
    power_db = 10.0 * np.log10(power / ref + 1e-6)
    return DEMONGram(frequencies_hz=f, times_s=t, power_db=power_db, config=config)