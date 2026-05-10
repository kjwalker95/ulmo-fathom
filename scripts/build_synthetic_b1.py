"""B1 spike runner: minimum-viable synthetic LOFAR clip from DeepShip ambient.

Per A1 §7 item 13 staged implementation. C1 will expand to NOAA + Watkins +
KRAKEN/BELLHOP IRs. B1 substitutes DeepShip vessel-free for NOAA NRS pending
acquisition (CEO direction 2026-05-10).
"""
from __future__ import annotations

from pathlib import Path

import click
import soundfile as sf
from rich.console import Console

from fathom.display.render import RenderConfig, render_lofar_gram
from fathom.grams.lofar import compute_lofar_gram
from fathom.models import LOFARConfig, StftConfig
from fathom.synthetic.generator import generate_b1_clip

CONSOLE = Console()


@click.command()
@click.option("--ambient-path", type=click.Path(exists=True, path_type=Path), required=True,
              help="Path to a DeepShip vessel-free recording.")
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("artifacts/sprint4_b1"))
@click.option("--frequency-hz", type=float, default=50.0)
@click.option("--t-start-s", type=float, default=2.0)
@click.option("--t-end-s", type=float, default=27.0)
@click.option("--snr-db", type=float, default=10.0)
@click.option("--seed", type=int, default=20260510)
def main(ambient_path, out_dir, frequency_hz, t_start_s, t_end_s, snr_db, seed):
    """Build one synthetic B1 clip + render its LOFAR gram for operator eyeball."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"b1_{ambient_path.stem}_{int(frequency_hz)}hz_{int(snr_db)}db"
    out_path = out_dir / f"{stem}.wav"

    CONSOLE.print(f"[cyan]Generating synthetic clip from {ambient_path.name}...[/cyan]")
    result = generate_b1_clip(
        ambient_path=ambient_path,
        out_path=out_path,
        frequency_hz=frequency_hz,
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        target_snr_db=snr_db,
        seed=seed,
    )

    wav, sr = sf.read(str(out_path), always_2d=False)
    cfg = LOFARConfig(
        stft=StftConfig(sample_rate=sr, n_fft=16384, hop_length=4096, window_length=16384),
        freq_min_hz=3.0, freq_max_hz=1000.0,
        normalization_train_window_bins=33, normalization_central_window_bins=5,
        normalization_gap_bins=1,
    )
    gram = compute_lofar_gram(wav.astype("float32"), cfg)
    render_cfg = RenderConfig(
        colormap="viridis", intensity_dynamic_range_db=50,
        figure_size_in=(12, 8), dpi=120,
        title=f"B1 synthetic: {stem}",
    )
    png_path = out_path.with_suffix(".png")
    render_lofar_gram(gram, png_path, render_cfg)

    CONSOLE.print(f"[green]Done.[/green]")
    CONSOLE.print(f"  WAV:      {result['wav_path']}")
    CONSOLE.print(f"  Manifest: {result['manifest_path']}")
    CONSOLE.print(f"  Audit:    {result['audit_path']}")
    CONSOLE.print(f"  LOFAR:    {png_path}")
    CONSOLE.print(f"")
    CONSOLE.print(f"[yellow]Operator eyeball:[/yellow] open {png_path}")
    CONSOLE.print(f"  Tonal at {frequency_hz} Hz should be visible from "
                  f"t={t_start_s}s to t={t_end_s}s, ~{snr_db} dB above ambient.")


if __name__ == "__main__":
    main()