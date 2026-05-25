"""Sprint 6 Cluster F — gram + ensemble-overlay + disagreement-overlay render.

Adapts scripts/sprint5_c6_render.py (single-model patch-tile inference at
stride=64, 75% overlap per A2) to the 5-member deep ensemble:
  - Per patch: 5 sigmoid masks -> per-pixel mean + per-pixel variance
  - Stitch via max-pool over overlapping patches (matches Sprint 5 C6 pattern)
  - Extract lines from stitched ensemble-mean via scipy.ndimage.label

Renders 3 PNGs per recording (under <out_dir>/<class>_<stem>/):
  gram.png                 LOFAR gram only (Convention B; Greys)
  ensemble_overlay.png     gram + ensemble-mean detected lines (red dashed)
  disagreement_overlay.png gram + member-disagreement variance heatmap
                           (Reds colormap, alpha=0.4, normalized to [0, 0.25])

Operator-facing framing (briefing text reused in any panel materials):
  warm/bright = ensemble disagreement is high = look-twice candidate
  dim/grey    = ensemble agrees (confident detection or confident absence)

Recruitment for the Sprint 6 panel deprioritized; this script ships
as Sprint 7+ panel infrastructure + Sprint 6 demonstration material
for the disagreement-overlay visual primitive.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from scipy.ndimage import label

from fathom.detection.ml_data import default_lofar_config
from fathom.detection.ml_train import build_model
from fathom.display.render import RenderConfig, render_lofar_gram
from fathom.grams.lofar import LOFARGram, compute_lofar_gram
from fathom.synthetic.ambient import load_deepship_ambient

CONSOLE = Console()

PATCH_SIZE = 256
STRIDE = 64
MIN_LINE_PIXELS = 16
BERNOULLI_MAX_VAR = 0.25  # max variance of Bernoulli(p=0.5)


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_member(
    checkpoint: Path, unet_base_channels: int, device: torch.device,
) -> torch.nn.Module:
    model = build_model(
        "unet", num_freq_bins=PATCH_SIZE, unet_base_channels=unet_base_channels,
    ).to(device)
    state = torch.load(str(checkpoint), map_location=device)
    state_dict = (
        state.get("model_state_dict", state) if isinstance(state, dict) else state
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _infer_ensemble_maps(
    models: list[torch.nn.Module],
    gram: LOFARGram,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Patch-tile ensemble inference at stride 64 (75% overlap).

    Returns (ensemble_mean, member_variance) both shape (n_freq, n_time),
    trimmed back to the original gram size after padded stitching.

    Stitching: max-pool over overlapping patches per pixel — matches the
    Sprint 5 C6 single-model pattern. Applied to BOTH the ensemble mean
    and the member variance independently. For the variance map this
    surfaces the most-contested overlapping-patch view of each pixel,
    which is what we want for the disagreement overlay.
    """
    gram_arr = gram.normalized_power_db
    n_freq, n_time = gram_arr.shape
    pad_freq = (
        (-(n_freq - PATCH_SIZE)) % STRIDE
        if n_freq > PATCH_SIZE else PATCH_SIZE - n_freq
    )
    pad_time = (
        (-(n_time - PATCH_SIZE)) % STRIDE
        if n_time > PATCH_SIZE else PATCH_SIZE - n_time
    )
    padded = np.pad(gram_arr, ((0, pad_freq), (0, pad_time)), mode="reflect")
    n_freq_p, n_time_p = padded.shape

    mean_max = np.zeros_like(padded, dtype=np.float32)
    var_max = np.zeros_like(padded, dtype=np.float32)

    f_starts = list(range(0, n_freq_p - PATCH_SIZE + 1, STRIDE))
    t_starts = list(range(0, n_time_p - PATCH_SIZE + 1, STRIDE))

    with torch.no_grad():
        for fs in f_starts:
            for ts in t_starts:
                patch = padded[fs:fs + PATCH_SIZE, ts:ts + PATCH_SIZE]
                tensor = (
                    torch.from_numpy(patch.astype(np.float32))
                    .unsqueeze(0).unsqueeze(0).to(device)
                )
                # Stack per-member sigmoid masks
                member_probs = np.stack(
                    [
                        torch.sigmoid(m(tensor)).squeeze().cpu().numpy()
                        for m in models
                    ],
                    axis=0,
                )  # (N, H, W)
                patch_mean = member_probs.mean(axis=0)
                patch_var = member_probs.var(axis=0)

                mean_max[fs:fs + PATCH_SIZE, ts:ts + PATCH_SIZE] = np.maximum(
                    mean_max[fs:fs + PATCH_SIZE, ts:ts + PATCH_SIZE],
                    patch_mean,
                )
                var_max[fs:fs + PATCH_SIZE, ts:ts + PATCH_SIZE] = np.maximum(
                    var_max[fs:fs + PATCH_SIZE, ts:ts + PATCH_SIZE],
                    patch_var,
                )

    return mean_max[:n_freq, :n_time], var_max[:n_freq, :n_time]


def _extract_lines(
    mask: np.ndarray,
    freqs_hz: np.ndarray,
    times_s: np.ndarray,
    *,
    bin_threshold: float,
    min_pixels: int = MIN_LINE_PIXELS,
) -> list[tuple[float, float, float]]:
    """Connected-component line extraction (Sprint 5 C6 pattern verbatim)."""
    binary = mask > bin_threshold
    labeled, n_components = label(binary)
    lines: list[tuple[float, float, float]] = []
    for k in range(1, n_components + 1):
        f_idx, t_idx = np.where(labeled == k)
        if len(f_idx) < min_pixels:
            continue
        freq = float(freqs_hz[f_idx].mean())
        t_start = float(times_s[t_idx.min()])
        t_end = float(times_s[t_idx.max()])
        lines.append((freq, t_start, t_end))
    return lines


def _save_disagreement_overlay(
    gram: LOFARGram,
    variance_map: np.ndarray,
    out_path: Path,
    config: RenderConfig,
    title: str,
) -> None:
    """Render gram in Greys + warm colormap variance overlay.

    Convention B: time vertical (oldest at top, newest at bottom),
    frequency horizontal. Matches render_lofar_gram axes exactly.

    Variance map normalized to [0, BERNOULLI_MAX_VAR=0.25] so the warm
    intensity is comparable across recordings (team review 2026-05-25).
    """
    img = gram.normalized_power_db
    f0 = float(gram.frequencies_hz[0])
    f1 = float(gram.frequencies_hz[-1])
    t0 = float(gram.times_s[0])
    t1 = float(gram.times_s[-1])

    fig, ax = plt.subplots(figsize=config.figure_size_in, dpi=config.dpi)
    vmax = float(np.percentile(img, 99))
    vmin = vmax - config.intensity_dynamic_range_db

    # Base gram: Greys, transposed for Convention B (time vertical).
    ax.imshow(
        img.T,
        aspect="auto",
        origin="upper",
        extent=(f0, f1, t1, t0),
        cmap=config.colormap,  # Greys per Sprint 3 convention
        vmin=vmin,
        vmax=vmax,
    )

    # Warm overlay: Reds colormap, alpha=0.4, fixed [0, 0.25] norm.
    ax.imshow(
        variance_map.T,
        aspect="auto",
        origin="upper",
        extent=(f0, f1, t1, t0),
        cmap=plt.cm.Reds,
        vmin=0.0,
        vmax=BERNOULLI_MAX_VAR,
        alpha=0.4,
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (s, relative to recording start; newest at bottom)")
    ax.set_title(title)

    # Colorbar for the variance overlay
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Reds,
        norm=plt.Normalize(vmin=0.0, vmax=BERNOULLI_MAX_VAR),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("ensemble member-disagreement variance")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=config.dpi)
    plt.close(fig)


@click.command()
@click.option(
    "--checkpoints", "checkpoints", multiple=True,
    type=click.Path(exists=True, path_type=Path), required=True,
    help="Ensemble member checkpoint paths (best.pt). Pass once per member.",
)
@click.option(
    "--recording", type=click.Path(exists=True, path_type=Path), required=True,
)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), required=True,
)
@click.option("--bin-threshold", type=float, default=0.001)
@click.option("--min-pixels", type=int, default=MIN_LINE_PIXELS)
@click.option("--unet-base-channels", type=int, default=64)
@click.option("--device", type=str, default="auto")
def main(
    checkpoints: tuple[Path, ...],
    recording: Path,
    output_dir: Path,
    bin_threshold: float,
    min_pixels: int,
    unet_base_channels: int,
    device: str,
) -> None:
    logging.getLogger("fathom").setLevel(logging.WARNING)
    device_obj = _autodetect_device() if device == "auto" else torch.device(device)

    parent_class = recording.parent.name
    stem = recording.stem
    rec_out_dir = output_dir / f"{parent_class}_{stem}"
    rec_out_dir.mkdir(parents=True, exist_ok=True)

    CONSOLE.print(f"[cyan]Recording: {recording}[/cyan]")
    CONSOLE.print(f"[cyan]Members ({len(checkpoints)}):[/cyan]")
    for cp in checkpoints:
        CONSOLE.print(f"  {cp}")
    CONSOLE.print(f"[cyan]Device: {device_obj}  bin_threshold={bin_threshold}[/cyan]")

    ambient, _sr = load_deepship_ambient(recording, target_sr=32_000)
    lofar_cfg = default_lofar_config()
    gram = compute_lofar_gram(ambient.astype("float32"), lofar_cfg)
    CONSOLE.print(
        f"Gram shape: {gram.normalized_power_db.shape}  "
        f"Duration: {gram.times_s[-1]:.1f}s"
    )

    models = [_load_member(cp, unet_base_channels, device_obj) for cp in checkpoints]

    CONSOLE.print("\n[cyan]Running ensemble inference + stitching...[/cyan]")
    ensemble_mean, member_variance = _infer_ensemble_maps(
        models, gram, device_obj,
    )
    CONSOLE.print(
        f"  ensemble_mean: max={ensemble_mean.max():.3f} mean={ensemble_mean.mean():.3f}"
    )
    CONSOLE.print(
        f"  member_variance: max={member_variance.max():.4f} "
        f"mean={member_variance.mean():.5f}"
    )

    lines = _extract_lines(
        ensemble_mean, gram.frequencies_hz, gram.times_s,
        bin_threshold=bin_threshold, min_pixels=min_pixels,
    )
    CONSOLE.print(f"  extracted {len(lines)} predicted lines")
    for i, (freq, t_start, t_end) in enumerate(sorted(lines)):
        CONSOLE.print(
            f"    line {i:3d}: f={freq:7.1f} Hz  t={t_start:6.1f}-{t_end:6.1f}s"
        )

    render_cfg = RenderConfig()

    gram_path = rec_out_dir / "gram.png"
    render_lofar_gram(gram, gram_path, render_cfg)
    CONSOLE.print(f"\n[green]Wrote {gram_path}[/green]")

    overlay_path = rec_out_dir / "ensemble_overlay.png"
    render_lofar_gram(gram, overlay_path, render_cfg, overlay_lines=lines)
    CONSOLE.print(f"[green]Wrote {overlay_path}[/green]")

    disagree_path = rec_out_dir / "disagreement_overlay.png"
    _save_disagreement_overlay(
        gram, member_variance, disagree_path, render_cfg,
        title=(
            f"{parent_class}/{stem} - ensemble disagreement "
            f"(warm = look-twice; grey = system agrees)"
        ),
    )
    CONSOLE.print(f"[green]Wrote {disagree_path}[/green]")


if __name__ == "__main__":
    main()