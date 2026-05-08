"""Sprint 2 false-alarm-rate sweep over the (peak_snr, persistence) grid.

Iterates the 3x3 grid configured under `far_sweep` in `configs/sprint2.yaml`
on the same per-class sanity sample (deterministic seed). Cells report
lines/hour aggregated across the sample. Output is `artifacts/sprint2_sanity/
far_sweep.md` by default — appended to the main sanity INDEX.md by C6.

Sample-rate: gram is computed once per recording; the inner loop varies only
peak_snr_threshold_db and min_persistence_s. TPSW + drift + gap_tolerance held
at config defaults.
"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import click
import soundfile as sf
import yaml
from rich.console import Console
from rich.logging import RichHandler

from fathom.detection import DetectionConfig, detect_lines
from fathom.events import EventBus
from fathom.grams.lofar import compute_lofar_gram
from fathom.ingestion.deepship import index_deepship
from fathom.models import LOFARConfig, StftConfig

logging.basicConfig(level=logging.WARNING, handlers=[RichHandler(rich_tracebacks=True)])
LOG = logging.getLogger("far_sweep")
CONSOLE = Console()

DEEPSHIP_RECORDING_START_UTC = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)


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


def _detection_config(cfg: dict, peak_snr_db: float, persistence_s: float) -> DetectionConfig:
    d = cfg["detection"]
    merge = d.get("merge", {})  # Sprint 3 added; absent in sprint2.yaml -> defaults False
    return DetectionConfig(
        tpsw_first_pass_threshold_db=d["tpsw"]["first_pass_threshold_db"],
        tpsw_min_unmasked_train_bins=d["tpsw"]["min_unmasked_train_bins"],
        peak_method=d["peaks"]["method"],
        peak_snr_threshold_db=peak_snr_db,
        peak_min_separation_time_bins=d["peaks"]["min_separation_time_bins"],
        peak_two_d_neighborhood=tuple(d["peaks"]["two_d_neighborhood"]),
        min_persistence_s=persistence_s,
        frequency_drift_bins=d["persistence"]["frequency_drift_bins"],
        gap_tolerance_time_bins=d["persistence"]["gap_tolerance_time_bins"],
        merge_nearby_lines=bool(merge.get("enabled", False)),
        merge_freq_tolerance_hz=merge.get("freq_tolerance_hz"),
    )


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("configs/sprint2.yaml"),
)
@click.option("--deepship-root", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--out-path",
    type=click.Path(path_type=Path),
    default=Path("artifacts/sprint2_sanity/far_sweep.md"),
)
@click.option("--n-per-class", type=int, default=5)
@click.option("--seed", type=int, default=20260520)
def main(
    config_path: Path,
    deepship_root: Path,
    out_path: Path,
    n_per_class: int,
    seed: int,
) -> None:
    """Run the 3x3 (peak_snr, persistence) FAR sweep on the per-class sanity sample."""
    random.seed(seed)
    cfg = yaml.safe_load(config_path.read_text())
    lofar_cfg = _build_lofar_config(cfg)

    snr_grid = list(cfg["far_sweep"]["peak_snr_threshold_db"])
    persistence_grid = list(cfg["far_sweep"]["min_persistence_s"])

    CONSOLE.print(f"indexing DeepShip at {deepship_root}")
    ds_index = index_deepship(deepship_root)
    by_class: dict[str, list] = {}
    for rec in ds_index.recordings:
        by_class.setdefault(rec.class_label or "Unknown", []).append(rec)

    sample = []
    for class_label in sorted(by_class):
        recs = by_class[class_label]
        sample.extend(random.sample(recs, min(n_per_class, len(recs))))

    grams: list[tuple[object, object]] = []
    total_duration_s = 0.0
    for rec in sample:
        wav, sr = sf.read(str(rec.path), always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != lofar_cfg.stft.sample_rate:
            CONSOLE.print(f"[yellow]skipping sample-rate mismatch: {rec.path}[/yellow]")
            continue
        gram = compute_lofar_gram(wav.astype("float32"), lofar_cfg)
        grams.append((rec, gram))
        total_duration_s += rec.duration_s

    sample_hours = total_duration_s / 3600
    CONSOLE.print(
        f"sample: {len(grams)} recordings, total duration {total_duration_s:.1f} s "
        f"({sample_hours:.3f} h)"
    )

    grid: dict[tuple[float, float], int] = {}
    for snr in snr_grid:
        for pers in persistence_grid:
            dcfg = _detection_config(cfg, snr, pers)
            total_lines = 0
            bus = EventBus()
            for rec, gram in grams:
                lines = detect_lines(
                    gram,
                    dcfg,
                    array_id="DEEPSHIP",
                    beam_id=None,
                    recording_start_utc=DEEPSHIP_RECORDING_START_UTC,
                    bus=bus,
                )
                total_lines += len(lines)
            grid[(snr, pers)] = total_lines
            CONSOLE.print(
                f"  snr={snr:g} dB, persistence={pers:g} s: "
                f"{total_lines} lines ({total_lines / sample_hours:.0f} lines/hour)"
            )

    md = [
        "# Sprint 2 FAR Sweep",
        "",
        f"Method: `{cfg['detection']['peaks']['method']}`. "
        f"Sample: {len(grams)} DeepShip recordings, total duration {sample_hours:.3f} h. "
        f"Cells = lines/hour aggregated across the sample.",
        "",
        "| peak_snr_db \\\\ persistence_s | "
        + " | ".join(f"{p:g} s" for p in persistence_grid)
        + " |",
        "|---|" + "---|" * len(persistence_grid),
    ]
    for snr in snr_grid:
        row = [f"**{snr:g} dB**"] + [
            f"{grid[(snr, pers)] / sample_hours:.0f}" for pers in persistence_grid
        ]
        md.append("| " + " | ".join(row) + " |")
    md += [
        "",
        "## Notes",
        "",
        "Real DeepShip recordings have rich broadband + harmonic content (machinery "
        "tonals, propeller cavitation, hull resonances) that produces many sustained "
        "peaks above any fixed threshold. The default operating point (8 dB, 3 s) "
        "yields very high lines/hour rates that operators cannot review individually. "
        "Sprint 3 will characterize FAR-vs-detection-rate at tightened thresholds; "
        "calibrated uncertainty (Phase 1; PCD v2 §5.1) will eventually replace fixed "
        "thresholds with per-line confidence with finite-sample coverage guarantees.",
        "",
        f"Raw line counts (not normalized): "
        + ", ".join(
            f"snr={snr:g}/pers={pers:g}: {grid[(snr, pers)]}"
            for snr in snr_grid
            for pers in persistence_grid
        )
        + ".",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))
    CONSOLE.print(f"[green]FAR sweep written to {out_path}[/green]")


if __name__ == "__main__":
    main()
