import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src.config import Config  # noqa: E402
from src.experiment import run_density_sweep  # noqa: E402
from src.plotting import plot_summary  # noqa: E402
from src.utils import ensure_dir, make_rng, mean, parse_density_list, seed_from_density_rep, std  # noqa: E402


def test_utils_helpers (local_tmp_path):
    out_dir = ensure_dir(local_tmp_path / "nested" / "dir")
    assert out_dir.exists()

    seed_a = seed_from_density_rep(100, 0.123, 5)
    seed_b = seed_from_density_rep(100, 0.123, 6)
    assert seed_a != seed_b

    rng1 = make_rng(42)
    rng2 = make_rng(42)
    assert rng1.integers(0, 100) == rng2.integers(0, 100)

    assert parse_density_list("0.1, 0.2,0.3") == [0.1, 0.2, 0.3]
    assert mean([]) == 0.0
    assert std([]) == 0.0
    assert mean([1.0, 3.0]) == 2.0
    assert std([1.0, 3.0]) == float(np.std([1.0, 3.0]))


def test_experiment_and_plotting (local_tmp_path):
    cfg = Config(
        N=5,
        densities=[0.2],
        burn_in_steps=0,
        measurement_steps=1,
        replications=1,
        output_dir=str(local_tmp_path / "results"),
    )

    df = run_density_sweep(cfg)
    summary_path = local_tmp_path / "results" / "summary.csv"
    assert summary_path.exists()
    assert list(df.columns) == [
        "density",
        "mean_speed",
        "std_speed",
        "mean_blocked",
        "std_blocked",
        "N",
        "A",
        "burn_in_steps",
        "measurement_steps",
        "K",
        "seed_base",
        "critical_density_est",
        "critical_drop_est",
    ]
    assert float(df["critical_density_est"].iloc[0]) >= float(df["density"].min())
    assert float(df["critical_density_est"].iloc[0]) <= float(df["density"].max())

    analysis_path = local_tmp_path / "results" / "analysis.json"
    assert analysis_path.exists()

    plot_dir = local_tmp_path / "results" / "plots"
    plot_summary(summary_path, plot_dir)

    assert (plot_dir / "speed_vs_density.png").exists()
    assert (plot_dir / "blocked_vs_density.png").exists()


def test_snapshot_output_created (local_tmp_path):
    cfg = Config(
        N=6,
        densities=[0.2],
        burn_in_steps=0,
        measurement_steps=3,
        replications=1,
        output_dir=str(local_tmp_path / "results"),
        save_snapshots=True,
        snapshot_densities=[0.2],
        snapshot_step="last",
    )

    run_density_sweep(cfg)

    snapshot_path = local_tmp_path / "results" / "snapshots" / "density_0.20.png"
    assert snapshot_path.exists()


def test_time_series_output_created (local_tmp_path):
    cfg = Config(
        N=6,
        densities=[0.2],
        burn_in_steps=0,
        measurement_steps=4,
        replications=2,
        output_dir=str(local_tmp_path / "results"),
        save_time_series=True,
        time_series_densities=[0.2],
        time_series_replication=1,
    )

    run_density_sweep(cfg)

    ts_path = local_tmp_path / "results" / "time_series" / "density_0.20_rep_1.csv"
    assert ts_path.exists()

    df = pd.read_csv(ts_path)
    assert list(df.columns) == ["timestep", "moved_count", "speed", "blocked_fraction"]
    assert len(df) == 4
