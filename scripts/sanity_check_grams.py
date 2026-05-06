"""Sprint 1 sanity-check script.

Indexes DeepShip, samples N recordings per class, computes LOFAR grams (and a
couple of DEMON grams for comparison), writes labeled PNGs with audit sidecars,
WAV clips alongside, and an INDEX.md operator-review checklist.
"""
from __future__ import annotations

import logging
import random
import shutil
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
from fathom.display.render import RenderConfig, render_demon_gram, render_lofar_gram
from fathom.events import Topic, get_default_bus
from fathom.grams.demon import compute_demon_gram
from fathom.grams.lofar import compute_lofar_gram
from fathom.ingestion.deepship import index_deepship, write_manifest
from fathom.models import (
    DEMONConfig,
    GramArtifact,
    GramType,
    LOFARConfig,
    StftConfig,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)])
LOG = logging.getLogger("sanity_check_grams")
CONSOLE = Console()


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


def _build_demon_config(cfg: dict) -> DEMONConfig:
    d = cfg["demon"]
    return DEMONConfig(
        sample_rate=d["sample_rate"],
        band_low_hz=d["band_low"],
        band_high_hz=d["band_high"],
        envelope_lpf_cutoff_hz=d["envelope_lpf_cutoff"],
        decimation_factor=d["decimation_factor"],
        n_fft=d["n_fft"],
        hop_length=d["hop_length"],
        window=d["window"],
    )


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=Path("configs/sprint1.yaml"))
@click.option("--deepship-root", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("artifacts/sprint1_sanity"))
@click.option("--n-per-class", type=int, default=5)
@click.option("--n-demon-examples", type=int, default=2)
@click.option("--seed", type=int, default=20260506)
def main(config_path, deepship_root, out_dir, n_per_class, n_demon_examples, seed):
    """Sprint 1 sanity-check: produce N LOFAR grams per DeepShip class."""
    random.seed(seed)
    cfg = _load_config(config_path)
    lofar_cfg = _build_lofar_config(cfg)
    demon_cfg = _build_demon_config(cfg)
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
    LOG.info("indexed %d recordings; manifest hash %s", len(ds_index.recordings), manifest_hash[:12])

    by_class: dict[str, list] = {}
    for rec in ds_index.recordings:
        by_class.setdefault(rec.class_label or "Unknown", []).append(rec)

    bus = get_default_bus()
    index_lines: list[str] = [
        "# Sprint 1 LOFAR Gram Sanity Check",
        "",
        "## Operator review checklist",
        "",
        "- [ ] Linear frequency axis (NOT mel-scale)",
        "- [ ] Time axis legible",
        "- [ ] Tonal lines visible against ambient",
        "- [ ] Color map matches operator intuition",
        "- [ ] Frequency range covers 1-1000 Hz primary view",
        "- [ ] Per-class samples are recognizable to operator memory",
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
                wav = wav.mean(axis=1)  # mono reduction (Sprint1_Plan §3)
            if sr != lofar_cfg.stft.sample_rate:
                LOG.warning(
                    "sample-rate mismatch %d vs %d for %s; skipping (resampling lands in a follow-up)",
                    sr, lofar_cfg.stft.sample_rate, rec.path,
                )
                continue
            gram = compute_lofar_gram(wav.astype("float32"), lofar_cfg)
            stem = f"{class_label}_{rec.recording_id}"
            png_path = out_dir / f"{stem}.png"
            wav_clip = out_dir / f"{stem}.wav"
            shutil.copy(rec.path, wav_clip)
            render_lofar_gram(gram, png_path, render_cfg)
            provenance = make_provenance(
                parameter_snapshot={"lofar": lofar_cfg.model_dump(), "render": render_cfg.__dict__},
                source_recording_path=rec.path,
                dataset_manifest_hash=manifest_hash,
            )
            write_audit_sidecar(png_path, provenance)
            artifact = GramArtifact(
                gram_type=GramType.LOFAR,
                image_path=png_path,
                sidecar_path=png_path.with_suffix(png_path.suffix + ".audit.json"),
                provenance=provenance,
                duration_s=rec.duration_s,
            )
            bus.publish(Topic.GRAM_GENERATED, artifact)
            index_lines.append(f"- {stem}: ![]({png_path.name}) [WAV]({wav_clip.name})")
        index_lines.append("")

    demon_pool = [r for class_recs in by_class.values() for r in class_recs]
    if demon_pool:
        demon_chosen = random.sample(demon_pool, min(n_demon_examples, len(demon_pool)))
        index_lines.append("## DEMON examples (light-touch validation)")
        index_lines.append("")
        for rec in demon_chosen:
            LOG.info("processing DEMON for %s/%s", rec.class_label, rec.recording_id)
            wav, sr = sf.read(str(rec.path), always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != demon_cfg.sample_rate:
                LOG.warning("sample-rate mismatch %d for DEMON of %s; skipping", sr, rec.path)
                continue
            dgram = compute_demon_gram(wav.astype("float32"), demon_cfg)
            stem = f"DEMON_{rec.class_label}_{rec.recording_id}"
            png_path = out_dir / f"{stem}.png"
            render_demon_gram(dgram, png_path, render_cfg)
            provenance = make_provenance(
                parameter_snapshot={"demon": demon_cfg.model_dump(), "render": render_cfg.__dict__},
                source_recording_path=rec.path,
                dataset_manifest_hash=manifest_hash,
            )
            write_audit_sidecar(png_path, provenance)
            index_lines.append(f"- {stem}: ![]({png_path.name})")
        index_lines.append("")

    (out_dir / "INDEX.md").write_text("\n".join(index_lines))
    CONSOLE.print(f"[green]Sanity-check artifacts written to {out_dir}/[/green]")


if __name__ == "__main__":
    main()