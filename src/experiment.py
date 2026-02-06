"""Experiment runner for density sweeps."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Config, save_config
from .plotting import plot_snapshot
from .simulation import run_simulation
from .utils import ensure_dir, make_seed

SUMMARY_COLUMNS = [
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
]


def run_density_sweep (config: Config) -> pd.DataFrame:
    """Run the density sweep and write summary CSV to output_dir."""
    output_dir = ensure_dir(config.output_dir)
    rows: list[dict[str, float | int]] = []

    resolved_densities = _densities(config.densities)
    config.densities = resolved_densities
    save_config(Path(output_dir) / "config_used.json", config)

    snapshot_density_keys = _density_keys(config.snapshot_densities)
    save_snapshots = bool(config.save_snapshots)
    snapshot_step = _resolve_snapshot_step(config.snapshot_step) if save_snapshots else None
    snapshot_dir = ensure_dir(Path(output_dir) / "snapshots") if save_snapshots else None

    time_series_density_keys = _density_keys(config.time_series_densities)
    save_time_series = bool(config.save_time_series)
    time_series_replication = int(config.time_series_replication)
    if save_time_series and time_series_replication < 0:
        raise ValueError("time_series_replication must be non-negative")
    if save_time_series and time_series_replication >= int(config.replications):
        raise ValueError("time_series_replication must be less than replications")
    time_series_dir = ensure_dir(Path(output_dir) / "time_series") if save_time_series else None

    for density in resolved_densities:
        per_run_v: list[float] = []
        per_run_b: list[float] = []
        A = int(np.floor(float(density) * config.N * config.N))
        should_snapshot = save_snapshots and (_density_key(density) in snapshot_density_keys)
        should_capture_time_series = save_time_series and (_density_key(density) in time_series_density_keys)

        for rep in range(int(config.replications)):
            seed = make_seed(config.seed_base, density, rep)
            run_snapshot_step = snapshot_step if (should_snapshot and rep == 0) else None
            run_collect_time_series = should_capture_time_series and (rep == time_series_replication)
            metrics = run_simulation(
                config,
                seed=seed,
                density=float(density),
                snapshot_step=run_snapshot_step,
                collect_time_series=run_collect_time_series,
            )
            per_run_v.append(float(metrics["mean_v"]))
            per_run_b.append(float(metrics["mean_b"]))

            if run_snapshot_step is not None and "snapshot_grid" in metrics and snapshot_dir is not None:
                snapshot_path = Path(snapshot_dir) / f"density_{float(density):.2f}.png"
                snapshot_grid = metrics["snapshot_grid"]
                if not isinstance(snapshot_grid, np.ndarray):
                    raise TypeError("snapshot_grid must be a numpy array")
                plot_snapshot(
                    snapshot_grid,
                    snapshot_path,
                    title=f"Density {float(density):.2f}",
                )

            if run_collect_time_series and "time_series" in metrics and time_series_dir is not None:
                time_series = metrics["time_series"]
                if not isinstance(time_series, list):
                    raise TypeError("time_series must be a list")
                time_series_df = pd.DataFrame(
                    time_series,
                    columns=["timestep", "moved_count", "speed", "blocked_fraction"],
                )
                time_series_path = (
                    Path(time_series_dir)
                    / f"density_{float(density):.2f}_rep_{int(time_series_replication)}.csv"
                )
                time_series_df.to_csv(time_series_path, index=False)

        mean_speed = float(np.mean(per_run_v)) if per_run_v else 0.0
        std_speed = float(np.std(per_run_v)) if per_run_v else 0.0
        mean_blocked = float(np.mean(per_run_b)) if per_run_b else 0.0
        std_blocked = float(np.std(per_run_b)) if per_run_b else 0.0

        rows.append(
            {
                "density": float(density),
                "mean_speed": mean_speed,
                "std_speed": std_speed,
                "mean_blocked": mean_blocked,
                "std_blocked": std_blocked,
                "N": int(config.N),
                "A": A,
                "burn_in_steps": int(config.burn_in_steps),
                "measurement_steps": int(config.measurement_steps),
                "K": int(config.replications),
                "seed_base": int(config.seed_base),
            }
        )

    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_path = Path(output_dir) / "summary.csv"
    df.to_csv(summary_path, index=False)
    return df


def _densities (values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def _density_key (value: float) -> int:
    return int(round(float(value) * 1000))


def _density_keys (values: Iterable[float]) -> set[int]:
    return {_density_key(v) for v in values}


def _resolve_snapshot_step (value: str | int) -> str | int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "last":
            return "last"
        if stripped.isdigit():
            return int(stripped)
    raise ValueError("snapshot_step must be 'last' or a non-negative integer")
