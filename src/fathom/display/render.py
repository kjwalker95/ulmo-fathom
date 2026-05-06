"""Operator-friendly LOFAR/DEMON gram rendering.

Linear frequency axis, time axis, intensity colormap with dynamic-range clipping.
Display conventions iterate with operator review (Sprint1_Plan §3, §6).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering for Docker and CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..grams.demon import DEMONGram  # noqa: E402
from ..grams.lofar import LOFARGram  # noqa: E402

LOG = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    colormap: str = "viridis"
    intensity_dynamic_range_db: float = 50.0
    figure_size_in: tuple[float, float] = (12.0, 8.0)
    dpi: int = 120
    title: str | None = None


def _save_imshow(
    img: np.ndarray,
    extent: tuple[float, float, float, float],
    out_path: Path,
    config: RenderConfig,
    ylabel: str,
    overlay_lines: list[tuple[float, float, float]] | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=config.figure_size_in, dpi=config.dpi)
    vmax = float(np.percentile(img, 99))
    vmin = vmax - config.intensity_dynamic_range_db
    ax.imshow(img, aspect="auto", origin="lower", extent=extent, cmap=config.colormap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Time (s, relative to recording start)")
    ax.set_ylabel(ylabel)
    if config.title:
        ax.set_title(config.title)
    if overlay_lines:
        for freq_hz, t0, t1 in overlay_lines:
            ax.hlines(freq_hz, t0, t1, colors="red", linestyles="--", linewidth=1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def render_lofar_gram(
    gram: LOFARGram,
    out_path: Path,
    config: RenderConfig,
    overlay_lines: list[tuple[float, float, float]] | None = None,
) -> Path:
    """Render a LOFAR gram to PNG.

    `overlay_lines` is a list of (frequency_hz, t_start_s, t_end_s) for line-of-interest
    overlays in Sprint 2+. Sprint 1 passes None.
    """
    img = gram.normalized_power_db
    extent = (
        float(gram.times_s[0]),
        float(gram.times_s[-1]),
        float(gram.frequencies_hz[0]),
        float(gram.frequencies_hz[-1]),
    )
    return _save_imshow(img, extent, Path(out_path), config, "Frequency (Hz)", overlay_lines)


def render_demon_gram(gram: DEMONGram, out_path: Path, config: RenderConfig) -> Path:
    img = gram.power_db
    extent = (
        float(gram.times_s[0]),
        float(gram.times_s[-1]),
        float(gram.frequencies_hz[0]),
        float(gram.frequencies_hz[-1]),
    )
    return _save_imshow(img, extent, Path(out_path), config, "Modulation frequency (Hz)")