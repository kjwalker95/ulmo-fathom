"""C1.3-lite visual smoke: generate same-seed no-prop and prop clips, render
both as LOFAR grams for operator eyeball ("would I mistake this for real?").
"""
from __future__ import annotations

from pathlib import Path

import soundfile as sf
import yaml

from fathom.display.render import RenderConfig, render_lofar_gram
from fathom.grams.lofar import compute_lofar_gram
from fathom.models import LOFARConfig, StftConfig
from fathom.synthetic.generator import generate_c1_1_clip
from fathom.synthetic.priors import PropagationGeometryPriors, TonalParameterPriors


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sprint3_cfg = yaml.safe_load((repo_root / "configs" / "sprint3.yaml").read_text())
    lofar_yaml = sprint3_cfg["lofar"]
    norm = lofar_yaml["normalization"]
    lofar_cfg = LOFARConfig(
        stft=StftConfig(
            sample_rate=lofar_yaml["sample_rate"],
            n_fft=lofar_yaml["n_fft"],
            hop_length=lofar_yaml["hop_length"],
            window_length=lofar_yaml["window_length"],
            window=lofar_yaml["window"],
        ),
        freq_min_hz=lofar_yaml["freq_min"],
        freq_max_hz=lofar_yaml["freq_max"],
        log_epsilon=lofar_yaml["log_epsilon"],
        normalization_train_window_bins=norm["train_window_bins"],
        normalization_central_window_bins=norm["central_window_bins"],
        normalization_gap_bins=norm["gap_bins"],
    )
    render_cfg = RenderConfig()  # operational defaults: Greys, 50 dB dynamic range

    ambient = next(Path("/Users/keith/Documents/data/DeepShip").rglob("*.wav"))
    print(f"ambient: {ambient}")

    out_dir = repo_root / "artifacts" / "c1_3_lite_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Force exactly 1 tonal source so side-by-side comparison is clean.
    priors_t = TonalParameterPriors(n_sources_distribution={1: 1.0})

    for label, prop in [("no_prop", None), ("prop", PropagationGeometryPriors())]:
        wav_path = out_dir / f"{label}.wav"
        result = generate_c1_1_clip(
            ambient_path=ambient,
            out_path=wav_path,
            seed=20260512,
            priors=priors_t,
            clip_duration_s=30.0,
            propagation_priors=prop,
        )
        wav, sr = sf.read(str(wav_path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        gram = compute_lofar_gram(wav.astype("float32"), lofar_cfg)
        png = out_dir / f"{label}.png"
        render_lofar_gram(gram, png, render_cfg)

        st = result["source_truths"][0]
        print(f"\n=== {label} ===")
        print(f"  f0_hz: {st['f0_hz']:.2f}")
        print(f"  target_snr_db: {st['target_snr_db']:.2f}")
        print(f"  pulses: {len(st['pulse_onsets_s'])}")
        print(f"  manifest version: {result['manifest'].generator_version}")
        print(f"  lines: {len(result['manifest'].lines)}")
        if "propagation_metadata" in st:
            pm = st["propagation_metadata"]
            print(
                f"  boost: {pm['source_level_boost_db']:.2f} dB | "
                f"gain_at_f0: {pm['channel_gain_at_f0_db']:.2f} dB"
            )
            geo = st["propagation_geometry"]
            print(
                f"  geometry: range={geo['horizontal_range_m']:.0f} m, "
                f"water={geo['water_depth_m']:.0f} m, "
                f"src_depth={geo['source_depth_m']:.1f} m, "
                f"recv_depth={geo['receiver_depth_m']:.1f} m"
            )
        print(f"  WAV: {wav_path}")
        print(f"  PNG: {png}")

    print(f"\nOpen both PNGs side-by-side:\n  open {out_dir}")


if __name__ == "__main__":
    main()