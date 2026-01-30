"""Experiment runner for density sweeps."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Config
from .simulation import run_simulation
from .utils import ensure_dir, seed_from_density_rep

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

    for density in _densities(config.densities):
        per_run_v: list[float] = []
        per_run_b: list[float] = []
        A = int(np.floor(float(density) * config.N * config.N))

        for rep in range(int(config.replications)):
            seed = seed_from_density_rep(config.seed_base, density, rep)
            metrics = run_simulation(config, seed=seed, density=float(density))
            per_run_v.append(float(metrics["mean_v"]))
            per_run_b.append(float(metrics["mean_b"]))

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
