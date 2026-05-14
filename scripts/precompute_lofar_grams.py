"""Pre-compute LOFAR grams for a synthetic training dataset.
Walks <data_dir>/*.wav and writes a `.lofar.npz` companion next to each WAV
containing `normalized_power_db` (2D float32) and `frequencies_hz` (1D
float32). SyntheticPatchDataset auto-detects these and skips on-the-fly STFT,
eliminating the dataloader bottleneck on A100 training (Sprint 5 A2,
2026-05-13).
Re-run any time the LOFARConfig changes — there is no config-hash check in
the payload, so callers must regenerate explicitly if STFT or normalization
parameters change.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import click
import numpy as np
import soundfile as sf
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from fathom.detection.ml_data import default_lofar_config
from fathom.grams.lofar import compute_lofar_gram
CONSOLE = Console()
def _compute_one(
    wav_path_str: str, sample_rate: int, force: bool
) -> tuple[str, str]:
    """Compute + save one clip's gram. Returns (path, status_string)."""
    wav_path = Path(wav_path_str)
    npz_path = wav_path.with_suffix(".lofar.npz")
    if npz_path.exists() and not force:
        return wav_path_str, "skipped"
    config = default_lofar_config(sample_rate=sample_rate)
    wav, sr = sf.read(str(wav_path), always_2d=False)
    if sr != sample_rate:
        return wav_path_str, f"sr-mismatch: file={sr} expected={sample_rate}"
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    gram = compute_lofar_gram(wav.astype("float32"), config)
    np.savez(
        npz_path,
        normalized_power_db=gram.normalized_power_db.astype("float32"),
        frequencies_hz=gram.frequencies_hz.astype("float32"),
    )
    return wav_path_str, "ok"
@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Dataset directory (recursively scanned for *.wav).",
)
@click.option(
    "--sample-rate",
    type=int,
    default=32000,
    help="Expected sample rate; gram config built around this. v1+v2 = 32 kHz.",
)
@click.option(
    "--num-workers",
    type=int,
    default=1,
    help="ProcessPoolExecutor workers. 1=serial; ~30 on A100, ~4-8 locally.",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="Re-compute even if .lofar.npz already exists.",
)
def main(
    data_dir: Path, sample_rate: int, num_workers: int, force: bool
) -> None:
    """Pre-compute LOFAR grams for all WAVs under --data-dir."""
    wavs = sorted(data_dir.rglob("*.wav"))
    if not wavs:
        raise click.UsageError(f"no .wav files under {data_dir}")
    CONSOLE.print(
        f"[cyan]{len(wavs)} WAVs under {data_dir}; workers={num_workers}[/cyan]"
    )
    stats = {"ok": 0, "skipped": 0, "error": 0}
    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task("Pre-computing", total=len(wavs))
        if num_workers <= 1:
            for wav in wavs:
                _, status = _compute_one(str(wav), sample_rate, force)
                if status == "ok":
                    stats["ok"] += 1
                elif status == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["error"] += 1
                    CONSOLE.print(f"[red]{wav.name}: {status}[/red]")
                progress.update(task, advance=1)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = [
                    pool.submit(_compute_one, str(w), sample_rate, force)
                    for w in wavs
                ]
                for fut in as_completed(futures):
                    wav_str, status = fut.result()
                    if status == "ok":
                        stats["ok"] += 1
                    elif status == "skipped":
                        stats["skipped"] += 1
                    else:
                        stats["error"] += 1
                        CONSOLE.print(f"[red]{Path(wav_str).name}: {status}[/red]")
                    progress.update(task, advance=1)
    CONSOLE.print(
        f"\n[green]Done. ok={stats['ok']} skipped={stats['skipped']} "
        f"errors={stats['error']}[/green]"
    )
if __name__ == "__main__":
    main()
