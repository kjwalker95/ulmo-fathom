"""C4 smoke: classical vs ML line-detection overlay on one DeepShip recording.

Renders a single LOFAR gram with three overlay sets:
  - classical loose: peak_snr=12 dB, persistence=8 s
  - classical tight: peak_snr=16 dB, persistence=20 s (frozen Phase 1 baseline)
  - ResNet-18 ML predictions from best.pt

Operator question this answers: does the ML detector find lines the
classical methods miss, and vice versa, on a real recording the model
has never seen?
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml
from matplotlib.patches import Patch
from rich.console import Console

from fathom.detection.lines import DetectionConfig, detect_lines
from fathom.detection.ml_eval import (
    PredictedLine,
    extract_predicted_lines_heatmap,
    extract_predicted_lines_mask,
)
from fathom.detection.ml_train import build_model
from fathom.events import EventBus
from fathom.grams.lofar import compute_lofar_gram
from fathom.ingestion._resample import resample_to
from fathom.models import LOFARConfig, StftConfig

CONSOLE = Console()
DEMO_RECORDING_START_UTC = datetime(2026, 5, 12, tzinfo=timezone.utc)


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def _build_detection_config(
    cfg: dict, peak_snr_db: float, persistence_s: float,
) -> DetectionConfig:
    d = cfg["detection"]
    merge = d.get("merge", {})
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


def _classical_overlays(
    gram, recording_id: str, detection_cfg: DetectionConfig,
) -> list[tuple[float, float, float]]:
    bus = EventBus()
    lines = detect_lines(
        gram, detection_cfg,
        array_id=recording_id, beam_id=None,
        recording_start_utc=DEMO_RECORDING_START_UTC, bus=bus,
    )
    out: list[tuple[float, float, float]] = []
    for loi in lines:
        t_start_s = (loi.timestamp - DEMO_RECORDING_START_UTC).total_seconds()
        out.append((loi.frequency_hz, t_start_s, t_start_s + loi.persistence_s))
    return out


def _ml_predictions(
    gram,
    ckpt_path: Path,
    architecture: str,
    device: torch.device,
    unet_base_channels: int = 64,
    stride: int = 64,
    batch_size: int = 32,
    class_threshold: float = 0.5,
    bin_threshold: float = 0.5,
) -> list[PredictedLine]:
    """Iterate 256x256 patches over the gram at inference stride, run model,
    collect heatmap predictions as PredictedLine in absolute (freq, time) coords.
    """
    model = build_model(
        architecture, num_freq_bins=256, unet_base_channels=unet_base_channels,
    ).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    img = gram.normalized_power_db.astype("float32")
    n_freq, n_time = img.shape
    ps = 256
    f_starts = list(range(0, n_freq - ps + 1, stride))
    if (n_freq - ps) % stride != 0:
        f_starts.append(n_freq - ps)
    t_starts = list(range(0, n_time - ps + 1, stride))
    if (n_time - ps) % stride != 0:
        t_starts.append(n_time - ps)

    addresses = [(f0, t0) for f0 in f_starts for t0 in t_starts]
    if not addresses:
        return []
    patches = np.stack(
        [img[f0:f0 + ps, t0:t0 + ps][None, :, :] for f0, t0 in addresses]
    )

    frame_duration_s = (
        float(gram.times_s[1] - gram.times_s[0]) if len(gram.times_s) > 1 else 0.0
    )

    predictions: list[PredictedLine] = []
    with torch.no_grad():
        for batch_start in range(0, len(patches), batch_size):
            batch = torch.from_numpy(
                patches[batch_start:batch_start + batch_size]
            ).to(device)
            batch_addresses = addresses[batch_start:batch_start + batch_size]

            if architecture == "resnet18":
                class_logit, heatmap_logit = model(batch)
                class_probs = torch.sigmoid(class_logit).cpu().numpy().reshape(-1)
                heatmap_probs = torch.sigmoid(heatmap_logit).cpu().numpy()
                for j, (f0, t0) in enumerate(batch_addresses):
                    predictions.extend(extract_predicted_lines_heatmap(
                        class_prob=float(class_probs[j]),
                        heatmap_probs=heatmap_probs[j],
                        patch_freq_axis_hz=gram.frequencies_hz[f0:f0 + ps],
                        patch_t_start_s=float(gram.times_s[t0]),
                        patch_t_end_s=float(
                            gram.times_s[min(t0 + ps - 1, n_time - 1)]
                        ),
                        class_threshold=class_threshold,
                        bin_threshold=bin_threshold,
                    ))
            elif architecture == "unet":
                mask_logit = model(batch)
                mask_probs_arr = torch.sigmoid(mask_logit).cpu().numpy()
                if mask_probs_arr.ndim == 4:
                    mask_probs_arr = mask_probs_arr[:, 0]
                for j, (f0, t0) in enumerate(batch_addresses):
                    predictions.extend(extract_predicted_lines_mask(
                        mask_probs=mask_probs_arr[j],
                        patch_freq_axis_hz=gram.frequencies_hz[f0:f0 + ps],
                        patch_t_start_s=float(gram.times_s[t0]),
                        patch_frame_duration_s=frame_duration_s,
                        bin_threshold=bin_threshold,
                    ))
            else:
                raise ValueError(f"unsupported architecture: {architecture}")
    return predictions


def _render_with_overlays(
    gram,
    classical_loose: list[tuple[float, float, float]],
    classical_tight: list[tuple[float, float, float]],
    ml_preds: list[PredictedLine],
    out_path: Path,
    title: str,
    ml_architecture_label: str = "ML",
) -> None:
    img = gram.normalized_power_db
    f0 = float(gram.frequencies_hz[0])
    f1 = float(gram.frequencies_hz[-1])
    t0 = float(gram.times_s[0])
    t1 = float(gram.times_s[-1])

    fig, ax = plt.subplots(figsize=(14, 9), dpi=120)
    vmax = float(np.percentile(img, 99))
    vmin = vmax - 50.0
    ax.imshow(
        img.T, aspect="auto", origin="upper",
        extent=(f0, f1, t1, t0),
        cmap="Greys", vmin=vmin, vmax=vmax,
    )

    for freq, ts, te in classical_loose:
        ax.plot([freq, freq], [ts, te], color="tab:blue", alpha=0.55, linewidth=1.0)
    for freq, ts, te in classical_tight:
        ax.plot([freq, freq], [ts, te], color="tab:red", alpha=0.9, linewidth=2.0)
    for pred in ml_preds:
        ax.plot(
            [pred.freq_hz, pred.freq_hz],
            [pred.t_start_s, pred.t_end_s],
            color="tab:orange", alpha=0.85, linewidth=2.5,
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (s, newest at bottom)")
    ax.set_title(title)
    ax.legend(handles=[
        Patch(color="tab:blue", alpha=0.55,
              label=f"classical loose 12 dB / 8 s  ({len(classical_loose)} lines)"),
        Patch(color="tab:red", alpha=0.9,
              label=f"classical tight 16 dB / 20 s  ({len(classical_tight)} lines)"),
        Patch(color="tab:orange", alpha=0.85,
              label=f"{ml_architecture_label} ML ({len(ml_preds)} unique lines)"),
    ], loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)

def _dedupe_predictions(
    preds: list[PredictedLine], freq_tol_hz: float = 5.0,
) -> list[PredictedLine]:
    """Collapse overlapping ML predictions across patches into unique lines.

    75%-overlap patch grid produces ~5-10 raw predictions per true tonal.
    Groups by freq within freq_tol_hz, merges each group's time extent
    (min..max) and confidence (max).
    """
    if not preds:
        return []
    preds_sorted = sorted(preds, key=lambda p: p.freq_hz)
    groups: list[list[PredictedLine]] = [[preds_sorted[0]]]
    for p in preds_sorted[1:]:
        if abs(p.freq_hz - groups[-1][0].freq_hz) <= freq_tol_hz:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [
        PredictedLine(
            freq_hz=float(np.mean([g.freq_hz for g in grp])),
            t_start_s=min(g.t_start_s for g in grp),
            t_end_s=max(g.t_end_s for g in grp),
            confidence=max(g.confidence for g in grp),
        )
        for grp in groups
    ]


@click.command()
@click.option(
    "--recording", type=click.Path(exists=True, path_type=Path), required=True,
    help="DeepShip / ShipsEar WAV.",
)
@click.option(
    "--ml-ckpt", type=click.Path(exists=True, path_type=Path),
    default=Path("artifacts/sprint4_baseline/resnet18_seed20260512/best.pt"),
)
@click.option(
    "--ml-architecture", type=click.Choice(["resnet18", "unet"]), default="resnet18",
)
@click.option(
    "--unet-base-channels", type=int, default=64,
    help="Must match the U-Net checkpoint's training (32 for our baseline run).",
)
@click.option(
    "--config", type=click.Path(exists=True, path_type=Path),
    default=Path("configs/sprint3.yaml"),
)
@click.option(
    "--out-dir", type=click.Path(path_type=Path),
    default=Path("artifacts/sprint4_c4_smoke"),
)
@click.option("--device", default="auto", help="auto|cpu|mps|cuda")
@click.option("--class-threshold", type=float, default=0.5,
              help="Class-head probability gate (training default 0.5; probe with 0.1 on real data).")
@click.option("--bin-threshold", type=float, default=0.5,
              help="Heatmap-bin probability threshold (training default 0.5).")
def main(
    recording: Path, ml_ckpt: Path, ml_architecture: str, unet_base_channels: int, config: Path, out_dir: Path, device: str,
    class_threshold: float, bin_threshold: float,
) -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    lofar_cfg = _build_lofar_config(cfg)
    device_obj = _autodetect_device() if device == "auto" else torch.device(device)

    CONSOLE.print(f"[cyan]Recording:[/cyan] {recording}")
    wav, source_sr = sf.read(str(recording), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype("float32")
    target_sr = lofar_cfg.stft.sample_rate
    if source_sr != target_sr:
        CONSOLE.print(f"  [yellow]resample {source_sr} -> {target_sr} Hz[/yellow]")
        wav = resample_to(wav, source_sr=int(source_sr), target_sr=target_sr)

    CONSOLE.print("[cyan]Computing LOFAR gram...[/cyan]")
    gram = compute_lofar_gram(wav, lofar_cfg)
    CONSOLE.print(
        f"  gram: {gram.normalized_power_db.shape}  duration: {gram.times_s[-1]:.1f}s"
    )

    recording_id = recording.stem

    CONSOLE.print("\n[cyan]Classical at (12 dB / 8 s)...[/cyan]")
    classical_loose = _classical_overlays(
        gram, recording_id, _build_detection_config(cfg, 12.0, 8.0),
    )
    CONSOLE.print(f"  {len(classical_loose)} lines")

    CONSOLE.print("[cyan]Classical at (16 dB / 20 s)...[/cyan]")
    classical_tight = _classical_overlays(
        gram, recording_id, _build_detection_config(cfg, 16.0, 20.0),
    )
    CONSOLE.print(f"  {len(classical_tight)} lines")

    CONSOLE.print(f"[cyan]{ml_architecture} inference (device={device_obj})...[/cyan]")
    ml_preds_raw = _ml_predictions(
        gram, ml_ckpt, ml_architecture, device_obj,
        unet_base_channels=unet_base_channels,
        class_threshold=class_threshold, bin_threshold=bin_threshold,
    )
    ml_preds = _dedupe_predictions(ml_preds_raw, freq_tol_hz=5.0)
    unique_freqs = sorted(round(p.freq_hz, 1) for p in ml_preds)
    CONSOLE.print(
        f"  {len(ml_preds_raw)} raw → {len(ml_preds)} unique lines after freq-dedup (±5 Hz)"
    )
    CONSOLE.print(f"  unique frequencies (Hz): {unique_freqs}")

    out_path = out_dir / f"{recording_id}_c4_overlays_{ml_architecture}.png"
    title = (
        f"C4 smoke: {recording_id}  |  classical 12/8 (blue) vs 16/20 (red) "
        f"vs {ml_architecture} best.pt (orange)"
    )
    _render_with_overlays(
        gram, classical_loose, classical_tight, ml_preds, out_path, title=title, ml_architecture_label=ml_architecture,
    )
    CONSOLE.print(f"\n[green]Wrote: {out_path}[/green]")


if __name__ == "__main__":
    main()