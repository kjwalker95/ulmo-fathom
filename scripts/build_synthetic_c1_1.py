"""C1.1 build runner: full A1 §3.3 parameterized synthetic LOFAR clips.

Produces N clips with weighted source-count sampling (including negatives),
decaying-cosine pulses, drift, and full per-frame truth manifests. Each clip
emits: WAV + truth manifest JSON + audit sidecar + LOFAR PNG with truth overlay.

PNGs use operational Convention B: frequency horizontal, time vertical
newest-at-bottom; Greys colormap (dark = energy). Red dashed overlays show
ground-truth freq_curve_hz per harmonic.
"""
from __future__ import annotations

from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from rich.console import Console  # noqa: E402

from fathom.grams.lofar import compute_lofar_gram  # noqa: E402
from fathom.models import (  # noqa: E402
    LOFARConfig,
    StftConfig,
    SyntheticTruthManifest,
)
from fathom.synthetic import (  # noqa: E402
    generate_c1_1_clip,
    stft_frame_times_s,
)

CONSOLE = Console()


def _list_ambient_wavs(ambient_dir: Path) -> list[Path]:
    return sorted(ambient_dir.rglob("*.wav"))


def _render_convention_b(
    wav_path: Path,
    manifest: SyntheticTruthManifest,
    out_path: Path,
    title: str,
) -> None:
    """Render LOFAR gram in operational Convention B with red truth overlay.

    - Frequency on X (horizontal), Time on Y (vertical, newest at bottom)
    - Greys colormap (dark = energy)
    - Red dashed lines: ground-truth freq_curve_hz per harmonic per source
    """
    wav, sr = sf.read(str(wav_path), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    cfg = LOFARConfig(
        stft=StftConfig(sample_rate=sr, n_fft=16384, hop_length=4096, window_length=16384),
        freq_min_hz=3.0,
        freq_max_hz=1000.0,
        normalization_train_window_bins=33,
        normalization_central_window_bins=5,
        normalization_gap_bins=1,
    )
    gram = compute_lofar_gram(wav.astype("float32"), cfg)

    img = gram.normalized_power_db.T  # rows = time, cols = freq

    fig, ax = plt.subplots(figsize=(8, 12), dpi=120)

    # extent=(freq_min, freq_max, time_max, time_min) with origin="upper"
    # gives newest-at-bottom (rows increment downward).
    extent = (
        float(gram.frequencies_hz[0]),
        float(gram.frequencies_hz[-1]),
        float(gram.times_s[-1]),
        float(gram.times_s[0]),
    )

    vmax = float(np.percentile(img, 99))
    vmin = vmax - 50.0

    ax.imshow(
        img,
        aspect="auto",
        origin="upper",
        extent=extent,
        cmap="Greys",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (s, newest at bottom)")
    ax.set_title(title, fontsize=10)

    # Truth overlays: red dashed line per harmonic-per-source.
    frame_times = stft_frame_times_s(len(wav), cfg.stft)
    for line in manifest.lines:
        if not line.freq_curve_hz or not line.mask_bin_indices:
            continue
        frame_indices = [fi for fi, _ in line.mask_bin_indices]
        if not frame_indices or max(frame_indices) >= len(frame_times):
            continue
        t_active = [float(frame_times[fi]) for fi in frame_indices]
        ax.plot(
            line.freq_curve_hz,
            t_active,
            color="red",
            linewidth=0.8,
            alpha=0.55,
            linestyle="--",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


@click.command()
@click.option(
    "--ambient-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory of ambient WAV recordings (recursively scanned).",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("artifacts/synthetic_c1_1"),
)
@click.option("--n-clips", type=int, default=5)
@click.option("--seed", type=int, default=20260510)
def main(ambient_dir: Path, out_dir: Path, n_clips: int, seed: int) -> None:
    """Build N C1.1 synthetic clips + render Convention-B LOFAR PNGs."""
    ambient_paths = _list_ambient_wavs(ambient_dir)
    if not ambient_paths:
        raise click.UsageError(f"no .wav files found under {ambient_dir}")
    CONSOLE.print(
        f"[cyan]Found {len(ambient_paths)} candidate ambient files under "
        f"{ambient_dir}[/cyan]"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    chooser_rng = np.random.default_rng(seed)
    chosen_indices = chooser_rng.choice(len(ambient_paths), size=n_clips, replace=True)

    summary = []
    for i, ambient_idx in enumerate(chosen_indices):
        ambient_path = ambient_paths[int(ambient_idx)]
        clip_seed = seed + i + 1
        clip_id = f"c1_1_seed{clip_seed}_{ambient_path.stem}"
        out_path = out_dir / f"{clip_id}.wav"

        CONSOLE.print(
            f"[cyan]Clip {i + 1}/{n_clips}: ambient={ambient_path.name}, "
            f"seed={clip_seed}[/cyan]"
        )
        result = generate_c1_1_clip(
            ambient_path=ambient_path,
            out_path=out_path,
            seed=clip_seed,
        )

        manifest: SyntheticTruthManifest = result["manifest"]
        png_path = out_path.with_suffix(".png")
        title = (
            f"{clip_id}  |  "
            f"n_sources={result['n_sources_realized']}/{result['n_sources_sampled']}  |  "
            f"negative={result['negative_label']}  |  "
            f"lines={len(manifest.lines)}"
        )
        _render_convention_b(out_path, manifest, png_path, title)

        summary.append({
            "clip_id": clip_id,
            "n_sources_sampled": result["n_sources_sampled"],
            "n_sources_realized": result["n_sources_realized"],
            "negative": result["negative_label"],
            "n_lines": len(manifest.lines),
            "wav": result["wav_path"],
            "png": png_path,
        })

    CONSOLE.print(f"\n[green]Done. {n_clips} clips written to {out_dir}[/green]")
    for s in summary:
        flag = "[NEG]" if s["negative"] else f"[POS x{s['n_sources_realized']}]"
        CONSOLE.print(
            f"  {flag:<10}  {s['n_lines']:>2} lines  {s['clip_id']}"
        )

    CONSOLE.print(
        f"\n[yellow]Operator eyeball:[/yellow] open {out_dir}/*.png"
    )
    CONSOLE.print(
        "  Convention B: freq horizontal, time vertical (newest at bottom); "
        "Greys (dark = energy)."
    )
    CONSOLE.print(
        "  Red dashed overlay = ground-truth freq_curve_hz per harmonic."
    )
    CONSOLE.print("  Look for:")
    CONSOLE.print("    - Source count visible: N distinct vertical stripes vs ambient texture")
    CONSOLE.print("    - Drift as tilt where |drift_rate| > 0.02 Hz/s")
    CONSOLE.print("    - Cluster modulation as horizontal banding when cluster_period < 10 s")
    CONSOLE.print("    - No broadband transients at pulse edges (cosine fade working)")
    CONSOLE.print("    - Red overlay tracks visible stripes within ~2 Hz")


if __name__ == "__main__":
    main()