"""TPSW normalization tests (Sprint 2 Cluster 5)."""
import numpy as np

from fathom.grams.normalization import split_window_normalize, tpsw_normalize


def _make_power_db_with_tonal(
    n_freq: int = 200,
    n_time: int = 400,
    tonal_bin: int = 100,
    tonal_db: float = 15.0,
    noise_std: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    power_db = rng.normal(0.0, noise_std, size=(n_freq, n_time))
    power_db[tonal_bin, :] += tonal_db
    return power_db


def test_tpsw_lower_ambient_near_strong_tonal():
    """At bins NEIGHBORING a strong tonal, single-pass ambient is biased high
    because the tonal sits inside their training ring. TPSW masks the tonal cell
    out of the second-pass ambient estimate, recovering an unbiased local ambient.

    AT the tonal's own bin, the central guard already excludes the tonal so
    single-pass already does the right thing — TPSW's win is at NEIGHBORS.
    """
    power_db = _make_power_db_with_tonal(tonal_bin=100, tonal_db=15.0)

    sp_residual = split_window_normalize(power_db.copy())
    tp_residual = tpsw_normalize(power_db.copy())

    sp_ambient = power_db - sp_residual
    tp_ambient = power_db - tp_residual

    neighbor_bins = list(range(95, 100)) + list(range(101, 106))
    sp_neighbor = float(sp_ambient[neighbor_bins].mean())
    tp_neighbor = float(tp_ambient[neighbor_bins].mean())

    # TPSW ambient at neighbors should be at least 0.1 dB cleaner (closer to 0).
    assert abs(tp_neighbor) < abs(sp_neighbor) - 0.1, (
        f"TPSW ambient near tonal ({tp_neighbor:+.3f} dB) is not cleaner than "
        f"single-pass ({sp_neighbor:+.3f} dB) by at least 0.1 dB"
    )


def test_tpsw_falls_back_when_overmasked_no_nan_or_inf():
    """When most cells in the train ring exceed the first-pass threshold,
    fewer than `min_unmasked_train_bins` cells remain unmasked and TPSW falls
    back to the first-pass ambient. The output must remain finite everywhere.
    """
    rng = np.random.default_rng(0)
    n_freq, n_time = 80, 80
    power_db = rng.normal(0.0, 1.0, size=(n_freq, n_time))
    # Inject strong tonals at every other frequency bin -> 50% above-threshold.
    power_db[::2, :] += 20.0

    out = tpsw_normalize(
        power_db,
        first_pass_threshold_db=6.0,
        min_unmasked_train_bins=16,
    )

    assert np.isfinite(out).all(), "TPSW produced NaN or Inf with overmasked input"


def test_tpsw_close_to_single_pass_when_no_strong_tonals():
    """With pure noise (no candidate-signal cells exceeding the threshold), the
    second-pass mask is empty and TPSW reduces to single-pass."""
    rng = np.random.default_rng(0)
    power_db = rng.normal(0.0, 1.0, size=(150, 150))

    sp_out = split_window_normalize(power_db.copy())
    tp_out = tpsw_normalize(power_db.copy(), first_pass_threshold_db=6.0)

    assert np.allclose(sp_out, tp_out, atol=0.5), (
        "TPSW diverges from single-pass on pure noise input "
        f"(max abs diff {np.abs(sp_out - tp_out).max():.3f} dB)"
    )