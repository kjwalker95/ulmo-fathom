"""Training run persistence: checkpoints, metrics CSV, loss-curve plots.

Per-run output structure (under <output_dir>/<architecture>_seed<seed>/):
  - config.json          — training run metadata snapshot (architecture, seed,
                            batch_size, n_samples_per_epoch, augmentation, etc.)
  - metrics.csv          — per-epoch row; rewritten after every epoch so a
                            killed run still has all completed epochs
  - best.pt              — checkpoint with lowest val_total seen
  - last.pt              — checkpoint at the current/final epoch
  - losses.png           — per-component train/val curves vs epoch
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


@dataclass
class TrainingRunMetadata:
    """Snapshot of the training run config; lives at config.json in the run dir."""
    architecture: str
    seed: int
    epochs: int
    batch_size: int
    n_samples_per_epoch: int
    val_fraction: float
    data_dir: str
    output_dir: str
    device: str
    learning_rate: float
    weight_decay: float
    augmentation: dict
    n_train_clips: int
    n_val_clips: int
    n_train_patches: int
    n_val_patches: int
    started_at_utc: str


def save_run_metadata(metadata: TrainingRunMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2))


def save_checkpoint(
    path: Path,
    *,
    model_state: dict,
    optimizer_state: dict,
    scheduler_state: dict,
    epoch: int,
    architecture: str,
    val_metric: float,
) -> None:
    """PyTorch checkpoint with model + optimizer + scheduler state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "epoch": epoch,
            "architecture": architecture,
            "val_metric": val_metric,
        },
        path,
    )


class MetricsLogger:
    """Per-epoch metrics CSV writer + train/val loss-curve plot.

    Tracks loss components from train_one_epoch + evaluate. CSV is rewritten
    on every `append` so a killed run still has all completed epochs.
    """

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def append(
        self,
        epoch: int,
        learning_rate: float,
        train_metrics: dict,
        val_metrics: dict,
    ) -> None:
        row: dict = {
            "epoch": epoch,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "learning_rate": float(learning_rate),
        }
        for k, v in train_metrics.items():
            if k == "n_batches":
                row["train_n_batches"] = int(v)
            else:
                row[f"train_{k}"] = float(v)
        for k, v in val_metrics.items():
            if k == "n_batches":
                row["val_n_batches"] = int(v)
            else:
                row[f"val_{k}"] = float(v)
        self.rows.append(row)
        self._flush()

    def _flush(self) -> None:
        # Union of all keys keeps the CSV stable if new components appear mid-run
        keys: list[str] = []
        for row in self.rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

    def plot_losses(self, png_path: Path) -> None:
        if not self.rows:
            return
        epochs = [r["epoch"] for r in self.rows]
        all_keys = set()
        for r in self.rows:
            all_keys.update(r.keys())
        train_keys = sorted(
            k for k in all_keys
            if k.startswith("train_") and k != "train_n_batches"
        )
        if not train_keys:
            return

        n = len(train_keys)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), dpi=100)
        if n == 1:
            axes = [axes]
        for ax, tkey in zip(axes, train_keys):
            component = tkey[len("train_"):]
            vkey = f"val_{component}"
            train_vals = [r.get(tkey) for r in self.rows]
            ax.plot(epochs, train_vals, label="train", marker="o")
            if vkey in all_keys:
                val_vals = [r.get(vkey) for r in self.rows]
                ax.plot(epochs, val_vals, label="val", marker="s")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(component)
            ax.set_title(component)
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path)
        plt.close(fig)