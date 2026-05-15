"""Render LOFAR grams + truth-line overlays for v2 (or v1) sample clips.

CEO eyeball workflow: pick N positive clips from a dataset's manifest.json
(seeded for reproducibility), render each clip's LOFAR gram with truth-line
overlays, save PNGs to an output directory. The CEO opens the PNGs visually.

Sprint 5 Cluster B acceptance #4: "operator eyeballs 5 random clip LOFAR
grams + truth overlays. Visual quality maintained despite wider priors."
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import click
import numpy as np
import soundfile as sf
from rich.console import Console

from fathom.detection.ml_data import default_lofar_config
from fathom.display.render import RenderConfig, render_lofar_gram
from fathom.grams.lofar import compute_lofar_gram
from fathom.models import SyntheticTruthManifest

CONSOLE = Console()


def _pick_positive_clips(manifest_path: Path, n: int, seed: int) -> list[dict]:
    m = json.loads(manifest_path.read_text())
    positives = [c for c in m["clips"] if not c["negative"]]
    rng = random.Random(seed)
    return rng.sample(positives, min(n, len(positives)))


def _build_overlays(
    manifest: SyntheticTruthManifest,
    freq_min_hz: float,
    freq_max_hz: float,
) -> list[tuple[float, float, float]]:
    """Build line-of-interest overlay tuples, filtered to the LOFAR gram band.

    Truth manifest carries every harmonic up to Nyquist; LOFAR gram only
    renders the configured band (3-1000 Hz by default). Out-of-band overlays
    would render as floating ghosts in the white space outside the gram. The
    underlying truth data still drives ml_eval (via mask_bin_indices), so
    this filter is purely a visualization concern.
    """
    overlays: list[tuple[float, float, float]] = []
    for line in manifest.lines:
        if not line.freq_curve_hz:
            continue
        freq_hz = float(np.mean(line.freq_curve_hz))
        if freq_hz < freq_min_hz or freq_hz > freq_max_hz:
            continue
        overlays.append((freq_hz, line.t_start_s, line.t_end_s))
    return overlays


@click.command()
@click.option(
    "--dataset-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Dataset directory (e.g. data/training_dataset_v2/).",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("artifacts/sprint5_v2_eyeball"),
)
@click.option("--n", type=int, default=5)
@click.option(
    "--seed",
    type=int,
    default=2026,
    help="Selection seed (same default as the spot-check one-liner).",
)
def main(dataset_dir: Path, out_dir: Path, n: int, seed: int) -> None:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise click.UsageError(f"missing manifest.json under {dataset_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    picks = _pick_positive_clips(manifest_path, n, seed)
    CONSOLE.print(
        f"[cyan]Rendering {len(picks)} positive clips from {dataset_dir} "
        f"(seed={seed})[/cyan]"
    )

    lofar_cfg = default_lofar_config()
    render_cfg = RenderConfig()

    for c in picks:
        clip_id = c["clip_id"]
        wav_path = dataset_dir / c["wav_relative_path"]
        truth_path = dataset_dir / c["manifest_relative_path"]

        wav, sr = sf.read(str(wav_path), always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        gram = compute_lofar_gram(wav.astype("float32"), lofar_cfg)

        truth = SyntheticTruthManifest.model_validate_json(truth_path.read_text())
        overlays = _build_overlays(truth, freq_min_hz=lofar_cfg.freq_min_hz,
            freq_max_hz=lofar_cfg.freq_max_hz,)
        png_path = out_dir / f"{clip_id}.png"
        render_lofar_gram(gram, png_path, render_cfg, overlay_lines=overlays)

        CONSOLE.print(
            f"  {clip_id:50s}  ambient={c['ambient_compound_key']:25s} "
            f"n_lines={c['n_lines']:2d}  png={png_path.name}"
        )

    CONSOLE.print(f"\n[green]Done. PNGs in {out_dir}/[/green]")


if __name__ == "__main__":
    main()