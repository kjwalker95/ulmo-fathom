"""Operator-friendly LOFAR/DEMON gram rendering — operational waterfall convention.

Convention B (operational IUSS): time on the vertical axis (oldest at top,
newest at bottom — waterfall scroll direction), frequency on the horizontal
axis. Tonal lines appear as vertical stripes (constant frequency, persistent
in time). The "static-TV-screen with vertical contact bars" display the
operator-side reads (CEO operator memory 2026-05-10).

Sprint 1's original implementation used Convention A (academic frequency-vs-
time framing — time horizontal, freq vertical, tonals horizontal). Corrected
2026-05-10 after CEO surfaced operational-display recall while reviewing B1
spike artifacts. Detection logic is orientation-agnostic (operates on
numerical (freq_bin, time_frame) arrays); only this rendering layer changes.
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
    colormap: str = "Greys"
    intensity_dynamic_range_db: float = 50.0
    figure_size_in: tuple[float, float] = (12.0, 8.0)
    dpi: int = 120
    title: str | None = None


def _save_imshow(
    img: np.ndarray,
    f0: float,
    f1: float,
    t0: float,
    t1: float,
    out_path: Path,
    config: RenderConfig,
    ylabel: str,
    overlay_lines: list[tuple[float, float, float]] | None = None,
) -> Path:
    """Render with Convention B: time vertical (oldest at top), freq horizontal.

    `img` is shape (n_freq, n_time) as produced by compute_lofar_gram /
    compute_demon_gram. We transpose to (n_time, n_freq) for display.

    `overlay_lines` is a list of (frequency_hz, t_start_s, t_end_s); each
    renders as a vertical red dashed line at frequency_hz from t_start to
    t_end on the time axis.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=config.figure_size_in, dpi=config.dpi)
    vmax = float(np.percentile(img, 99))
    vmin = vmax - config.intensity_dynamic_range_db
    # extent=(left, right, bottom, top) = (f0, f1, t1, t0) puts t=t0 at TOP
    # (oldest, where waterfall scrolls in from) and t=t1 at BOTTOM (newest).
    ax.imshow(
        img.T,
        aspect="auto",
        origin="upper",
        extent=(f0, f1, t1, t0),
        cmap=config.colormap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    if config.title:
        ax.set_title(config.title)
    if overlay_lines:
        for freq_hz, t_start, t_end in overlay_lines:
            ax.vlines(
                freq_hz, t_start, t_end,
                colors="red", linestyles="--", linewidth=1.0,
            )
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
    """Render a LOFAR gram to PNG in operational waterfall convention.

    `overlay_lines` is a list of (frequency_hz, t_start_s, t_end_s) for
    line-of-interest overlays in Sprint 2+.
    """
    img = gram.normalized_power_db
    f0 = float(gram.frequencies_hz[0])
    f1 = float(gram.frequencies_hz[-1])
    t0 = float(gram.times_s[0])
    t1 = float(gram.times_s[-1])
    return _save_imshow(
        img, f0, f1, t0, t1, Path(out_path), config,
        ylabel="Time (s, relative to recording start; newest at bottom)",
        overlay_lines=overlay_lines,
    )


def render_demon_gram(gram: DEMONGram, out_path: Path, config: RenderConfig) -> Path:
    img = gram.power_db
    f0 = float(gram.frequencies_hz[0])
    f1 = float(gram.frequencies_hz[-1])
    t0 = float(gram.times_s[0])
    t1 = float(gram.times_s[-1])
    return _save_imshow(
        img, f0, f1, t0, t1, Path(out_path), config,
        ylabel="Time (s, relative to recording start; newest at bottom)",
    )