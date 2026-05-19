"""Render LOFAR grams + ML detection overlays for the operator-recognition test.

Sprint 5 Cluster C6 (PCD v4 §13.1 primary success criterion). Per input
recording, produces two PNGs under --out-dir:
  - <class>_<stem>_gram.png       Clean LOFAR gram, no overlays.
  - <class>_<stem>_ml_lines.png   Gram + extracted predicted lines as
                                  dashed verticals (Convention B).

CEO workflow:
  1. Open <stem>_gram.png. Mentally note (or write down) the lines an
     IUSS analyst would flag at the watch floor on this recording.
  2. Open <stem>_ml_lines.png. See what ML detected at the C3 winning
     combo (ratio=0.75, threshold=0.001).
  3. Compare per recording: matches, ML misses, manual misses, ML false
     positives. Write the operator-recognition verdict.

Line extraction: connected-component analysis on the post-threshold
binary mask via scipy.ndimage.label. Each blob → one (mean_freq,
t_start, t_end) tuple. Components below --min-pixels are filtered as
noise.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
from scipy.ndimage import label

from fathom.detection.ml_data import default_lofar_config
from fathom.detection.ml_train import build_model
from fathom.display.render import RenderConfig, render_lofar_gram
from fathom.grams.lofar import LOFARGram, compute_lofar_gram
from fathom.synthetic.ambient import load_deepship_ambient


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _infer_recording_mask(
    *,
    model,
    gram: LOFARGram,
    device: torch.device,
    patch_size: int = 256,
    stride: int = 64,
) -> np.ndarray:
    """Patch-tile inference over the full-recording gram at 75% overlap (A2
    design memo inference stride). Aggregate overlapping patch predictions
    via max-pooling across patches that cover each pixel.

    Returns (n_freq, n_time) sigmoid-probability mask in [0, 1].
    """
    gram_arr = gram.normalized_power_db
    n_freq, n_time = gram_arr.shape

    pad_freq = (-(n_freq - patch_size)) % stride if n_freq > patch_size else patch_size - n_freq
    pad_time = (-(n_time - patch_size)) % stride if n_time > patch_size else patch_size - n_time
    padded = np.pad(gram_arr, ((0, pad_freq), (0, pad_time)), mode="reflect")
    n_freq_p, n_time_p = padded.shape

    pred_max = np.zeros_like(padded, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        f_starts = list(range(0, n_freq_p - patch_size + 1, stride))
        t_starts = list(range(0, n_time_p - patch_size + 1, stride))
        for fs in f_starts:
            for ts in t_starts:
                patch = padded[fs:fs + patch_size, ts:ts + patch_size]
                tensor = (
                    torch.from_numpy(patch.astype(np.float32))
                    .unsqueeze(0).unsqueeze(0).to(device)
                )
                logits = model(tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                pred_max[fs:fs + patch_size, ts:ts + patch_size] = np.maximum(
                    pred_max[fs:fs + patch_size, ts:ts + patch_size], prob
                )

    return pred_max[:n_freq, :n_time]


def _extract_lines_via_connected_components(
    mask: np.ndarray,
    freqs_hz: np.ndarray,
    times_s: np.ndarray,
    *,
    bin_threshold: float,
    min_pixels: int = 16,
) -> list[tuple[float, float, float]]:
    """Connected-component extraction on the binary mask. One line per blob,
    represented as (mean_freq_hz, t_start_s, t_end_s)."""
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


@click.command()
@click.option(
    "--recording",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="DeepShip recording (.wav).",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    default=Path(
        "artifacts/sprint5_ratio_sweep/unet_seed20260513_ratio0.75/best.pt"
    ),
    help="Model checkpoint. Default: lowest-ECE seed at C3 winning ratio.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("artifacts/sprint5_c6"),
)
@click.option(
    "--bin-threshold", type=float, default=0.001,
    help="C3 winning operational threshold.",
)
@click.option(
    "--min-pixels", type=int, default=16,
    help="Min connected-component size (pixels) to extract as a line.",
)
@click.option("--unet-base-channels", type=int, default=64)
@click.option("--device", type=str, default="auto")
def main(
    recording: Path,
    checkpoint: Path,
    out_dir: Path,
    bin_threshold: float,
    min_pixels: int,
    unet_base_channels: int,
    device: str,
) -> None:
    """Render LOFAR gram + ML overlay for the operator-recognition test."""
    logging.getLogger("fathom").setLevel(logging.WARNING)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_obj = _autodetect_device() if device == "auto" else torch.device(device)

    print(f"Recording: {recording}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device_obj}")
    print(f"Threshold: {bin_threshold}")

    ambient, sr = load_deepship_ambient(recording, target_sr=32_000)
    lofar_cfg = default_lofar_config()
    gram = compute_lofar_gram(ambient.astype("float32"), lofar_cfg)
    print(f"Gram shape: {gram.normalized_power_db.shape}")
    print(f"Duration: {gram.times_s[-1]:.1f} s")

    state = torch.load(str(checkpoint), map_location=device_obj)
    state_dict = (
        state.get("model_state_dict", state) if isinstance(state, dict) else state
    )
    model = build_model(
        "unet", num_freq_bins=256, unet_base_channels=unet_base_channels,
    ).to(device_obj)
    model.load_state_dict(state_dict)
    print("Loaded checkpoint")

    print("Running inference...")
    mask = _infer_recording_mask(model=model, gram=gram, device=device_obj)
    print(f"Mask max: {mask.max():.3f}  mean: {mask.mean():.3f}")

    lines = _extract_lines_via_connected_components(
        mask, gram.frequencies_hz, gram.times_s,
        bin_threshold=bin_threshold,
        min_pixels=min_pixels,
    )
    print(f"Extracted {len(lines)} predicted lines (connected components)")
    for i, (freq, t_start, t_end) in enumerate(sorted(lines)):
        print(f"  line {i:3d}: f={freq:7.1f} Hz  t={t_start:6.1f}-{t_end:6.1f} s")

    stem = recording.stem
    parent_class = recording.parent.name
    out_stem = f"{parent_class}_{stem}"

    render_cfg = RenderConfig()
    render_lofar_gram(gram, out_dir / f"{out_stem}_gram.png", render_cfg)
    print(f"Wrote {out_dir / f'{out_stem}_gram.png'}")

    render_lofar_gram(
        gram, out_dir / f"{out_stem}_ml_lines.png",
        render_cfg, overlay_lines=lines,
    )
    print(f"Wrote {out_dir / f'{out_stem}_ml_lines.png'}")


if __name__ == "__main__":
    main()