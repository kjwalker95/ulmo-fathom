"""Sprint 3 single-command Tuor line-detection demo.

Takes a recording path (DeepShip or ShipsEar), runs the Tuor pipeline at the
Sprint 3 operating point (`configs/sprint3.yaml` defaults), renders a labeled
LOFAR gram with detected lines overlaid, and prints a one-line summary.
Targets <30 s on a typical laptop.

Phase 0 exit gate per Phase0_Plan.md §6: laptop demo works end-to-end on at
least three DeepShip recordings and at least three ShipsEar recordings.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import soundfile as sf
import yaml
from rich.console import Console
from rich.logging import RichHandler

from fathom.audit import make_provenance, write_audit_sidecar
from fathom.detection import DetectionConfig, detect_lines
from fathom.display.render import RenderConfig, render_lofar_gram
from fathom.events import EventBus
from fathom.grams.lofar import compute_lofar_gram
from fathom.ingestion._resample import resample_to
from fathom.models import LOFARConfig, StftConfig

logging.basicConfig(level=logging.WARNING, handlers=[RichHandler(rich_tracebacks=True)])
LOG = logging.getLogger("sprint3_demo")
CONSOLE = Console()

DEMO_RECORDING_START_UTC = datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)


def _build_lofar_config(cfg: dict) -> LOFARConfig:
    lofar = cfg["lofar"]
    norm = lofar["normalization"]
    return LOFARConfig(
        stft=StftConfig(
            sample_rate=lofar["sample_rate"],
            n_fft=lofar["n_fft"],
            hop_length=lofar["hop_length"],
            window_length=lofar["window_length"],
            window=lofar["window"],
        ),
        freq_min_hz=lofar["freq_min"],
        freq_max_hz=lofar["freq_max"],
        log_epsilon=lofar["log_epsilon"],
        normalization_train_window_bins=norm["train_window_bins"],
        normalization_central_window_bins=norm["central_window_bins"],
        normalization_gap_bins=norm["gap_bins"],
    )


def _build_detection_config(cfg: dict) -> DetectionConfig:
    d = cfg["detection"]
    merge = d.get("merge", {})
    return DetectionConfig(
        tpsw_first_pass_threshold_db=d["tpsw"]["first_pass_threshold_db"],
        tpsw_min_unmasked_train_bins=d["tpsw"]["min_unmasked_train_bins"],
        peak_method=d["peaks"]["method"],
        peak_snr_threshold_db=d["peaks"]["snr_threshold_db"],
        peak_min_separation_time_bins=d["peaks"]["min_separation_time_bins"],
        peak_two_d_neighborhood=tuple(d["peaks"]["two_d_neighborhood"]),
        min_persistence_s=d["persistence"]["min_persistence_s"],
        frequency_drift_bins=d["persistence"]["frequency_drift_bins"],
        gap_tolerance_time_bins=d["persistence"]["gap_tolerance_time_bins"],
        merge_nearby_lines=bool(merge.get("enabled", False)),
        merge_freq_tolerance_hz=merge.get("freq_tolerance_hz"),
    )


@click.command()
@click.argument("recording", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("configs/sprint3.yaml"),
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("artifacts/sprint3_demo"),
)
def main(recording: Path, config_path: Path, out_dir: Path) -> None:
    """Tuor demo: WAV in -> LOFAR + detected lines + labeled gram out."""
    t0 = time.time()
    cfg = yaml.safe_load(config_path.read_text())
    lofar_cfg = _build_lofar_config(cfg)
    detection_cfg = _build_detection_config(cfg)
    render_cfg = RenderConfig(
        colormap=cfg["display"]["colormap"],
        intensity_dynamic_range_db=cfg["display"]["intensity_dynamic_range_db"],
        figure_size_in=tuple(cfg["display"]["figure_size_in"]),
        dpi=cfg["display"]["dpi"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    wav, source_sr = sf.read(str(recording), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype("float32")
    target_sr = lofar_cfg.stft.sample_rate
    if source_sr != target_sr:
        CONSOLE.print(f"[yellow]resampling {source_sr} Hz -> {target_sr} Hz[/yellow]")
        wav = resample_to(wav, source_sr=int(source_sr), target_sr=target_sr)

    gram = compute_lofar_gram(wav, lofar_cfg)
    bus = EventBus()
    lines = detect_lines(
        gram,
        detection_cfg,
        array_id="DEMO",
        beam_id=None,
        recording_start_utc=DEMO_RECORDING_START_UTC,
        bus=bus,
    )

    overlays: list[tuple[float, float, float]] = []
    for loi in lines:
        t_start_s = (loi.timestamp - DEMO_RECORDING_START_UTC).total_seconds()
        t_end_s = t_start_s + loi.persistence_s
        overlays.append((loi.frequency_hz, t_start_s, t_end_s))

    png_path = out_dir / f"{recording.stem}.png"
    render_lofar_gram(gram, png_path, render_cfg, overlay_lines=overlays)

    parameter_snapshot = {
        "lofar": lofar_cfg.model_dump(),
        "detection": {
            "tpsw_first_pass_threshold_db": detection_cfg.tpsw_first_pass_threshold_db,
            "tpsw_min_unmasked_train_bins": detection_cfg.tpsw_min_unmasked_train_bins,
            "peak_method": detection_cfg.peak_method,
            "peak_snr_threshold_db": detection_cfg.peak_snr_threshold_db,
            "peak_min_separation_time_bins": detection_cfg.peak_min_separation_time_bins,
            "peak_two_d_neighborhood": list(detection_cfg.peak_two_d_neighborhood),
            "min_persistence_s": detection_cfg.min_persistence_s,
            "frequency_drift_bins": detection_cfg.frequency_drift_bins,
            "gap_tolerance_time_bins": detection_cfg.gap_tolerance_time_bins,
            "merge_nearby_lines": detection_cfg.merge_nearby_lines,
            "merge_freq_tolerance_hz": detection_cfg.merge_freq_tolerance_hz,
        },
        "render": render_cfg.__dict__,
        "source_sample_rate_hz": int(source_sr),
        "resampled_to_hz": target_sr if source_sr != target_sr else None,
    }
    provenance = make_provenance(
        parameter_snapshot=parameter_snapshot,
        source_recording_path=recording,
    )
    write_audit_sidecar(png_path, provenance)

    elapsed = time.time() - t0
    top_3 = sorted(lines, key=lambda L: L.snr_db, reverse=True)[:3]
    summary_top = ", ".join(f"{L.frequency_hz:.0f} Hz @ {L.snr_db:.1f} dB" for L in top_3)
    CONSOLE.print(
        f"[green]Tuor demo: {len(lines)} line(s) in {elapsed:.1f}s. "
        f"Top: {summary_top or 'none'}. -> {png_path}[/green]"
    )


if __name__ == "__main__":
    main()
