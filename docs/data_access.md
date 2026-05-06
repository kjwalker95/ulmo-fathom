# Data access

Sprint 1 uses two unclassified passive-acoustic datasets: DeepShip (primary) and ShipsEar (optional in Sprint 1, required by Sprint 3). Neither is committed to the repo. Both live outside the working tree.

## DeepShip

**Source:** Northwestern Polytechnical University. Released 2021. Citation: Irfan, Zhu, Meng, Iqbal, "DeepShip: An underwater acoustic benchmark dataset and a separable convolution based autoencoder for classification."

**Size:** ~47 hours, 265 distinct ships across 4 classes (Cargo, Tanker, Tug, Passengership). ~50 GB on disk.

**Access:**
1. The smaller portion is available on the project GitHub: https://github.com/irfankamboh/DeepShip
2. The remainder requires emailing `mirfan@mail.nwpu.edu.cn` and accepting research-use terms.

**Layout on disk (this release):** Flat per-class. Each numeric `.wav` file is a distinct vessel.
```
deepship/
├── README.txt
├── Cargo/
│   ├── 15.wav
│   ├── 27.wav
│   └── ...
├── Passengership/
├── Tanker/
└── Tug/
```

The Sprint 1 plan §3 originally anticipated a `<class>/<vessel>/<recording>.wav` layout; the actual release is flatter. Because each numeric `.wav` is a distinct vessel (DeepShip has 265 ships and the per-class numeric IDs are unique), the loader (`src/fathom/ingestion/deepship.py`) treats `recording_id == vessel_id` for this layout. Vessel-level metadata is preserved, and Phase 1's vessel-level holdout splits work correctly.

**Local path (this dev machine):** `/Users/keith/Documents/data/deepship/`

**Sample rate:** 32,000 Hz. Use as the LOFAR `sample_rate` default in `configs/sprint1.yaml`.

## ShipsEar

**Source:** University of Vigo. Released 2016. Citation: Santos-Domínguez, Torres-Guijarro, Cardenal-López, Pena-Gimenez, "ShipsEar: An underwater vessel noise database."

**Size:** ~3 GB, 90 recordings, 11 vessel types, port of Vigo, Spain.

**Access:** Email the authors at the University of Vigo. Research-use terms apply. The release format (filename convention, directory layout) needs to be confirmed once the data lands; the loader (`src/fathom/ingestion/shipsear.py`) uses a best-guess regex on filenames and warns when the pattern does not match.

**Sample rate:** 52,734 Hz. Resampling to 32 kHz is an ingestion-layer responsibility (planned for a follow-up sprint; Sprint 1 sanity check skips ShipsEar files at the wrong sample rate with a warning).

## Storage policy

- Raw audio lives **outside** the repo. `.gitignore` blocks `*.wav`, `*.flac`, `*.aif`, `*.aiff`, and the `data/` directory.
- Manifests (`*.json` produced by `write_manifest`) are also gitignored under `artifacts/`. They are reproducible from the dataset.
- Audit sidecars (`*.audit.json`) are produced under `artifacts/` per run and gitignored.

## What this doc does not cover

- Tier 2 data sources (academic partnerships, NOAA cooperative data, self-collected hydrophone deployments) — addressed in Phase 3 plan when relevant.
- Tier 3 classified data sources (IUSS archival, sonobuoy, MAD, ACINT) — accessed via SBIR/CRADA pathway with NUWC; covered in proposal materials, not in this engineering doc.
