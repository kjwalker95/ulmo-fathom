"""DCLDE 2018 LF annotation → biological-confuser clip library.

ETL one-off: scans annotation CSVs (LF full split), maps each quality-filtered
annotation row to its source FLAC + offset, slices [t_start - pad, t_end + pad]
at native 2 kHz, writes per-clip mono WAV. Builds manifest.json conforming to
the BiologicalClipLibrary schema (consumed by fathom.synthetic.biologicals).

DCLDE-specific. Watkins / other future biological sources will get their own
extraction scripts producing the same manifest schema.
"""
from __future__ import annotations

import csv
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import soundfile as sf
from rich.console import Console

from fathom.models import BiologicalClip, BiologicalClipLibrary

CONSOLE = Console()
LOG = logging.getLogger(__name__)

SOURCE_DATASET = "dclde_2018"

# Species code → display name + LF frequency band (Hz).
# Bm = Balaenoptera musculus (blue whale): ~17-20 Hz tonal
# Eg = Eubalaena glacialis (North Atlantic right whale): ~50-200 Hz up-call
SPECIES_TABLE: dict[str, dict] = {
    "Bm": {"name": "blue_whale", "freq_range_hz": (10.0, 30.0)},
    "Eg": {"name": "north_atlantic_right_whale", "freq_range_hz": (50.0, 200.0)},
}


def _parse_file_start_timestamp(filename: str) -> datetime:
    """e.g. 'HAT_A_02_121021_000000.d100.x.flac' -> 2012-10-21T00:00:00 UTC."""
    parts = filename.split("_")
    yymmdd = parts[3]
    hhmmss = parts[4].split(".")[0]
    yy = int(yymmdd[:2])
    year = 2000 + yy if yy < 70 else 1900 + yy
    return datetime(
        year, int(yymmdd[2:4]), int(yymmdd[4:6]),
        int(hhmmss[0:2]), int(hhmmss[2:4]), int(hhmmss[4:6]),
        tzinfo=timezone.utc,
    )


def _scan_audio_root(audio_root: Path) -> list[dict]:
    """Return one entry per FLAC: {path, start_ts, end_ts, sample_rate, dep_dir}."""
    entries = []
    for dep_dir in sorted(audio_root.glob("dclde_2018_*_lf")):
        audio_subdir = dep_dir / "audio"
        if not audio_subdir.is_dir():
            continue
        for flac_path in sorted(audio_subdir.glob("*.flac")):
            info = sf.info(str(flac_path))
            start_ts = _parse_file_start_timestamp(flac_path.name)
            end_ts = datetime.fromtimestamp(
                start_ts.timestamp() + info.frames / info.samplerate,
                tz=timezone.utc,
            )
            entries.append({
                "path": flac_path,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "sample_rate": info.samplerate,
                "dep_dir": dep_dir.name,
            })
    return entries


def _read_annotation_csv(csv_path: Path) -> list[dict]:
    """Cols: project, site, species_code, t_start_iso, t_end_iso, quality."""
    rows = []
    with csv_path.open("r") as f:
        for parts in csv.reader(f):
            if len(parts) < 6:
                continue
            try:
                t_start = datetime.fromisoformat(parts[3].replace("Z", "+00:00"))
                t_end = datetime.fromisoformat(parts[4].replace("Z", "+00:00"))
            except ValueError:
                continue
            # DCLDE 2018 LF row timestamps lack a tz suffix; spec is UTC
            # (deploymentInfo.txt LF_Start uses Z). Force-attach UTC.
            if t_start.tzinfo is None:
                t_start = t_start.replace(tzinfo=timezone.utc)
            if t_end.tzinfo is None:
                t_end = t_end.replace(tzinfo=timezone.utc)
            rows.append({
                "project": parts[0],
                "site": parts[1],
                "species_code": parts[2],
                "t_start": t_start,
                "t_end": t_end,
                "quality": parts[5],
            })
    return rows


def _find_file(entries: list[dict], t_start: datetime, t_end: datetime) -> dict | None:
    """File containing [t_start, t_end) in full. Returns None if span crosses boundary."""
    for e in entries:
        if e["start_ts"] <= t_start and t_end < e["end_ts"]:
            return e
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@click.command()
@click.option("--annotations-root", type=click.Path(exists=True, path_type=Path), required=True,
              help="DCLDE 2018 products/detections root.")
@click.option("--audio-root", type=click.Path(exists=True, path_type=Path), required=True,
              help="DCLDE 2018 audio root (downloaded LF deployments).")
@click.option("--out-dir", type=click.Path(path_type=Path),
              default=Path("/Users/keith/Documents/data/dclde_2018_clips"))
@click.option("--pad-s", type=float, default=0.5,
              help="Pre/post pad on each side of annotation.")
@click.option("--quality-keep", multiple=True, default=("good",),
              help="Quality values to retain; repeat for multiple.")
def main(annotations_root: Path, audio_root: Path, out_dir: Path,
         pad_s: float, quality_keep: tuple[str, ...]) -> None:
    """Extract DCLDE 2018 LF biological clips."""
    quality_keep_set = set(quality_keep)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_entries = _scan_audio_root(audio_root)
    CONSOLE.print(
        f"[cyan]Scanned {len(audio_entries)} FLAC files across "
        f"{len({e['dep_dir'] for e in audio_entries})} deployments[/cyan]"
    )

    # LF "full" CSVs only (full = dev + eval superset; avoids duplication).
    csv_paths = sorted(annotations_root.rglob("*_LF_full.csv"))
    CONSOLE.print(f"[cyan]Processing {len(csv_paths)} LF annotation files[/cyan]")

    clips: list[BiologicalClip] = []
    n_skipped_quality = n_skipped_no_audio = n_skipped_unknown_species = 0
    unknown_species: dict[str, int] = {}

    for csv_path in csv_paths:
        rows = _read_annotation_csv(csv_path)
        n_in_file = 0
        for row in rows:
            if row["quality"] not in quality_keep_set:
                n_skipped_quality += 1
                continue
            if row["species_code"] not in SPECIES_TABLE:
                n_skipped_unknown_species += 1
                unknown_species[row["species_code"]] = unknown_species.get(row["species_code"], 0) + 1
                continue
            entry = _find_file(audio_entries, row["t_start"], row["t_end"])
            if entry is None:
                n_skipped_no_audio += 1
                continue

            sr = entry["sample_rate"]
            offset_s = max(0.0, (row["t_start"] - entry["start_ts"]).total_seconds() - pad_s)
            ann_dur_s = (row["t_end"] - row["t_start"]).total_seconds()
            duration_s = ann_dur_s + 2.0 * pad_s
            n_samples = int(duration_s * sr)
            start_frame = int(offset_s * sr)

            with sf.SoundFile(str(entry["path"])) as f:
                f.seek(start_frame)
                clip_audio = f.read(n_samples, dtype="float32", always_2d=False)
            if len(clip_audio) < n_samples:
                clip_audio = np.pad(clip_audio, (0, n_samples - len(clip_audio)))

            species_info = SPECIES_TABLE[row["species_code"]]
            site_dir = out_dir / row["species_code"] / row["site"]
            site_dir.mkdir(parents=True, exist_ok=True)
            clip_id = f"{row['project']}_{row['site']}_{row['species_code']}_{len(clips):05d}"
            clip_path = site_dir / f"{clip_id}.wav"
            sf.write(str(clip_path), clip_audio, samplerate=sr, subtype="PCM_16")

            clips.append(BiologicalClip(
                clip_id=clip_id,
                source_dataset=SOURCE_DATASET,
                species_code=row["species_code"],
                species_name=species_info["name"],
                site=row["site"],
                deployment=entry["dep_dir"],
                sample_rate_hz=sr,
                duration_s=float(duration_s),
                pad_s=float(pad_s),
                annotated_t_start_s=float(pad_s),
                annotated_t_end_s=float(pad_s + ann_dur_s),
                freq_range_hz=species_info["freq_range_hz"],
                quality=row["quality"],
                sha256=_sha256(clip_path),
                relative_path=str(clip_path.relative_to(out_dir)),
            ))
            n_in_file += 1

        CONSOLE.print(f"  {csv_path.name}: extracted {n_in_file}/{len(rows)} rows")

    species_counts: dict[str, int] = {}
    for c in clips:
        species_counts[c.species_code] = species_counts.get(c.species_code, 0) + 1

    library = BiologicalClipLibrary(
        library_id=f"{SOURCE_DATASET}_lf_v1",
        source_dataset=SOURCE_DATASET,
        n_clips=len(clips),
        species_counts=species_counts,
        clips=clips,
        built_at=datetime.now(timezone.utc),
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(library.model_dump_json(indent=2))

    CONSOLE.print(f"\n[green]Done. Extracted {len(clips)} clips → {out_dir}[/green]")
    CONSOLE.print(f"  manifest: {manifest_path}")
    for sp, n in sorted(species_counts.items()):
        CONSOLE.print(f"  {sp} ({SPECIES_TABLE[sp]['name']}): {n}")
    CONSOLE.print(f"  skipped (quality not in {sorted(quality_keep_set)}): {n_skipped_quality}")
    CONSOLE.print(f"  skipped (no audio coverage): {n_skipped_no_audio}")
    CONSOLE.print(f"  skipped (unknown species): {n_skipped_unknown_species}")
    if unknown_species:
        CONSOLE.print(f"    unknown species seen: {dict(sorted(unknown_species.items()))}")


if __name__ == "__main__":
    main()