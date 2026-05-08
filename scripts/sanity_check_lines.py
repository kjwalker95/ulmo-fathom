"""Tuor detection sanity-check script.

Loads N DeepShip recordings per class, computes a LOFAR gram, runs
`detect_lines` against the configured detection parameters, renders the gram
with detected-line overlays, writes per-recording `*.lines.jsonl` with audit
sidecars, and produces an INDEX.md operator-review checklist. Pass
`--config` to switch sprint configs (e.g., `configs/sprint3.yaml` for the
tightened Sprint 3 operating point with cluster-merge enabled).

Recording-start UTC is a placeholder for unclassified DeepShip data — the
dataset has no real Z-time. The constant `DEEPSHIP_RECORDING_START_UTC`
anchors `LineOfInterest.timestamp` deterministically so container parity
holds across runs.
"""
from __future__ import annotations

import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import click
import soundfile as sf
import yaml
from rich.console import Console
from rich.logging import RichHandler

from fathom.audit import (
    hash_file_sha256,
    make_provenance,
    write_audit_sidecar,
)
from fathom.detection import DetectionConfig, detect_lines
from fathom.display.render import RenderConfig, render_lofar_gram
from fathom.events import get_default_bus
from fathom.grams.lofar import compute_lofar_gram
from fathom.ingestion.deepship import index_deepship, write_manifest
from fathom.models import LOFARConfig, StftConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
LOG = logging.getLogger("sanity_check_lines")
CONSOLE = Console()

# Deterministic recording-start UTC anchor for unclassified DeepShip data.
DEEPSHIP_RECORDING_START_UTC = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


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
    merge = d.get("merge", {})  # Sprint 3 added; absent in sprint2.yaml -> defaults False
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
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("configs/sprint2.yaml"),
)
@click.option("--deepship-root", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("artifacts/sprint2_sanity"))
@click.option("--n-per-class", type=int, default=5)
@click.option("--seed", type=int, default=20260520)
@click.option(
    "--peak-method",
    type=click.Choice(["per_bin", "two_d"]),
    default=None,
    help="Override detection.peaks.method (used for C6 per-bin vs 2D ablation).",
)
def main(
    config_path: Path,
    deepship_root: Path,
    out_dir: Path,
    n_per_class: int,
    seed: int,
    peak_method: str | None,
) -> None:
    """Tuor sanity check: LOFAR grams with line-of-interest overlays per DeepShip class."""
    random.seed(seed)
    cfg = _load_config(config_path)
    if peak_method is not None:
        cfg["detection"]["peaks"]["method"] = peak_method
    lofar_cfg = _build_lofar_config(cfg)
    detection_cfg = _build_detection_config(cfg)
    render_cfg = RenderConfig(
        colormap=cfg["display"]["colormap"],
        intensity_dynamic_range_db=cfg["display"]["intensity_dynamic_range_db"],
        figure_size_in=tuple(cfg["display"]["figure_size_in"]),
        dpi=cfg["display"]["dpi"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config_snapshot.yaml")

    LOG.info("indexing DeepShip at %s", deepship_root)
    ds_index = index_deepship(deepship_root)
    manifest_path = out_dir / "deepship_manifest.json"
    write_manifest(ds_index, manifest_path)
    manifest_hash = hash_file_sha256(manifest_path)
    LOG.info(
        "indexed %d recordings; manifest hash %s",
        len(ds_index.recordings),
        manifest_hash[:12],
    )

    by_class: dict[str, list] = {}
    for rec in ds_index.recordings:
        by_class.setdefault(rec.class_label or "Unknown", []).append(rec)

    bus = get_default_bus()

    index_lines: list[str] = [
        "# Tuor Detection Sanity Check",
        "",
        f"Method: `{detection_cfg.peak_method}`. Full parameters in `config_snapshot.yaml`.",
        "",
        "## Operator review checklist",
        "",
        "- [ ] Detected lines (red dashed overlays) visually correspond to tonal stripes",
        "- [ ] False alarms (overlays where no tonal stripe exists) are operationally manageable",
        "- [ ] Sub-threshold misses (visible tonal stripes with no overlay) noted in review",
        "- [ ] Per-class line counts non-zero and class-plausible",
        "",
        "## Recordings",
        "",
    ]

    for class_label in sorted(by_class):
        recs = by_class[class_label]
        chosen = random.sample(recs, min(n_per_class, len(recs)))
        index_lines.append(f"### {class_label}")
        index_lines.append("")
        for rec in chosen:
            LOG.info("processing %s/%s", class_label, rec.recording_id)
            wav, sr = sf.read(str(rec.path), always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)  # mono reduction
            if sr != lofar_cfg.stft.sample_rate:
                LOG.warning(
                    "sample-rate mismatch %d vs %d for %s; skipping (resampling pathway is Phase 1)",
                    sr,
                    lofar_cfg.stft.sample_rate,
                    rec.path,
                )
                continue
            gram = compute_lofar_gram(wav.astype("float32"), lofar_cfg)
            lines = detect_lines(
                gram,
                detection_cfg,
                array_id="DEEPSHIP",
                beam_id=None,
                recording_start_utc=DEEPSHIP_RECORDING_START_UTC,
                bus=bus,
            )

            stem = f"{class_label}_{rec.recording_id}"
            png_path = out_dir / f"{stem}.png"
            jsonl_path = out_dir / f"{stem}.lines.jsonl"

            # Each overlay is (freq_hz, t_start_s, t_end_s); times derived from
            # LineOfInterest.timestamp + persistence_s relative to the deterministic
            # recording-start anchor.
            overlays: list[tuple[float, float, float]] = []
            for loi in lines:
                t_start_s = (loi.timestamp - DEEPSHIP_RECORDING_START_UTC).total_seconds()
                t_end_s = t_start_s + loi.persistence_s
                overlays.append((loi.frequency_hz, t_start_s, t_end_s))
            render_lofar_gram(gram, png_path, render_cfg, overlay_lines=overlays)

            jsonl_path.write_text(
                "\n".join(loi.model_dump_json() for loi in lines) + ("\n" if lines else "")
            )

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
                },
                "render": render_cfg.__dict__,
            }
            provenance = make_provenance(
                parameter_snapshot=parameter_snapshot,
                source_recording_path=rec.path,
                dataset_manifest_hash=manifest_hash,
            )
            write_audit_sidecar(png_path, provenance)
            write_audit_sidecar(jsonl_path, provenance)

            index_lines.append(
                f"- {stem}: ![]({png_path.name}) [lines.jsonl]({jsonl_path.name}) "
                f"({len(lines)} line{'s' if len(lines) != 1 else ''})"
            )
        index_lines.append("")

    (out_dir / "INDEX.md").write_text("\n".join(index_lines))
    CONSOLE.print(f"[green]Tuor sanity-check artifacts written to {out_dir}/[/green]")


if __name__ == "__main__":
    main()
