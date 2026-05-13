"""Vessel-level split manifest builder (Sprint 3 Cluster 3).

Produces train/val/test vessel ID partitions for a given dataset, stratified
by class (default), with a deterministic seed. Output is `<out_dir>/<dataset>_splits.json`
plus a SHA256 sidecar.

Per PCD v3 §12.2 / CLAUDE.md architectural binding: vessel-level holdout is
enforced; recording-level splits leak the published-DeepShip-baselines kind of
leakage we explicitly do not compete on.

The DeepShip flat-layout convention treats `recording_id` as `vessel_id`
(per `src/fathom/ingestion/deepship.py` and the dataset README). Phase 1 training
reads the resulting manifest by joining vessel IDs back to the dataset index;
the manifest itself is never re-derived from raw data once built.

Sprint 3 ships DeepShip splits; ShipsEar splits are a one-line follow-up in
Phase 1 once the ShipsEar resampling pathway is exercised at scale.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from random import Random

import click
from rich.console import Console
from rich.logging import RichHandler

from fathom.audit import hash_file_sha256, now_utc
from fathom.ingestion.deepship import index_deepship
from fathom.models import SplitManifest

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)])
LOG = logging.getLogger("build_splits")
CONSOLE = Console()


def _compound_key(class_label: str, vessel_id: str) -> str:
    """`<class>/<vessel_id>` compound key disambiguating DeepShip's flat layout.

    DeepShip class folders share bare numeric filenames (`Cargo/41.wav`,
    `Passengership/41.wav`, `Tanker/41.wav` are three different physical ships).
    The compound key keeps each (class, vessel_id) tuple uniquely addressable in
    the manifest. Downstream consumers parse via `class_label, stem = key.split("/", 1)`.
    """
    return f"{class_label}/{vessel_id}"


def _stratified_partition(
    vessels_by_class: dict[str, list[str]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Per-class shuffle then partition; aggregate as compound keys across classes.

    Within each class, vessels are sorted (deterministic) then shuffled with the
    master Random instance, which is iterated over `sorted(vessels_by_class)` for
    deterministic class order. Counts are floor-rounded for train/val; test
    receives the residual so all vessels land in some split.

    Returns compound-key entries (`<class>/<vessel_id>`) so cross-class bare-ID
    collisions stay disambiguated (see `_compound_key`).
    """
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    rng = Random(seed)
    for class_label in sorted(vessels_by_class):
        vessels = sorted(vessels_by_class[class_label])
        rng.shuffle(vessels)
        n = len(vessels)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        train.extend(_compound_key(class_label, v) for v in vessels[:n_train])
        val.extend(_compound_key(class_label, v) for v in vessels[n_train : n_train + n_val])
        test.extend(_compound_key(class_label, v) for v in vessels[n_train + n_val :])
        LOG.info(
            "class %s: %d vessels -> train=%d val=%d test=%d",
            class_label,
            n,
            n_train,
            n_val,
            n_test,
        )
    return sorted(train), sorted(val), sorted(test)


def _flat_partition(
    vessels_by_class: dict[str, list[str]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Unstratified shuffle-then-split. Used when --no-stratify is set.

    Aggregates compound keys across classes before shuffling so cross-class bare-ID
    collisions stay disambiguated. The shuffle ignores class boundaries (that's the
    point of --no-stratify) but each entry remains uniquely identifiable.
    """
    rng = Random(seed)
    all_keys = sorted(
        _compound_key(cls, v)
        for cls, vs in vessels_by_class.items()
        for v in vs
    )
    rng.shuffle(all_keys)
    n = len(all_keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        sorted(all_keys[:n_train]),
        sorted(all_keys[n_train : n_train + n_val]),
        sorted(all_keys[n_train + n_val :]),
    )


@click.command()
@click.option("--deepship-root", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("artifacts/sprint3_splits"),
)
@click.option("--seed", type=int, default=20260520)
@click.option("--train-ratio", type=float, default=0.70)
@click.option("--val-ratio", type=float, default=0.15)
@click.option(
    "--stratify-by-class/--no-stratify",
    default=True,
    help="Stratify by class so each split has ~proportional class distribution (default: on).",
)
def main(
    deepship_root: Path,
    out_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    stratify_by_class: bool,
) -> None:
    """Build a vessel-level train/val/test split manifest for DeepShip."""
    test_ratio = 1.0 - train_ratio - val_ratio
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise click.BadParameter("train + val + test ratios must sum to 1.0")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise click.BadParameter("ratios must be non-negative")

    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("indexing DeepShip at %s", deepship_root)
    ds_index = index_deepship(deepship_root)

    by_class: dict[str, set[str]] = defaultdict(set)
    for rec in ds_index.recordings:
        if rec.vessel_id is None:
            LOG.warning("recording %s has no vessel_id; skipping", rec.recording_id)
            continue
        by_class[rec.class_label or "Unknown"].add(rec.vessel_id)
    vessels_by_class = {cls: sorted(vs) for cls, vs in by_class.items()}
    total_vessels = sum(len(vs) for vs in vessels_by_class.values())
    LOG.info(
        "indexed %d unique vessels across %d classes",
        total_vessels,
        len(vessels_by_class),
    )

    if stratify_by_class:
        train, val, test = _stratified_partition(
            vessels_by_class,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    else:
        train, val, test = _flat_partition(
            vessels_by_class,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

    manifest = SplitManifest(
        dataset="deepship",
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratified_by_class=stratify_by_class,
        train_vessels=train,
        val_vessels=val,
        test_vessels=test,
        built_at=now_utc(),
        notes=f"Built from {len(ds_index.recordings)} recordings at {deepship_root}",
    )

    out_path = out_dir / "deepship_splits.json"
    out_path.write_text(manifest.model_dump_json(indent=2))
    digest = hash_file_sha256(out_path)
    sha_path = out_path.with_suffix(out_path.suffix + ".sha256")
    sha_path.write_text(digest + "\n")
    LOG.info("wrote %s (sha256: %s)", out_path, digest[:12])

    CONSOLE.print(
        f"[green]Splits: train={len(train)} val={len(val)} test={len(test)} "
        f"(stratified={stratify_by_class}, seed={seed})[/green]"
    )
    CONSOLE.print(f"[green]Manifest: {out_path}[/green]")


if __name__ == "__main__":
    main()
