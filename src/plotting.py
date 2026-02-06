"""Plotting helpers for experiment summaries."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def plot_summary (summary_csv_path: str | Path, output_dir: str | Path) -> None:
    """Generate plots from summary CSV."""
    df = pd.read_csv(summary_csv_path)
    output_dir = ensure_dir(output_dir)

    _plot_errorbar(
        df,
        x="density",
        y="mean_speed",
        yerr="std_speed",
        title="Mean Speed vs Density",
        ylabel="Mean speed",
        output_path=Path(output_dir) / "speed_vs_density.png",
    )
    _plot_errorbar(
        df,
        x="density",
        y="mean_blocked",
        yerr="std_blocked",
        title="Mean Blocked Fraction vs Density",
        ylabel="Mean blocked fraction",
        output_path=Path(output_dir) / "blocked_vs_density.png",
    )


def _plot_errorbar (
        df: pd.DataFrame,
        *,
        x: str,
        y: str,
        yerr: str,
        title: str,
        ylabel: str,
        output_path: Path,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        df[x],
        df[y],
        yerr=df[yerr],
        fmt="o-",
        capsize=3,
    )
    plt.xlabel("Density")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_snapshot (
        grid: np.ndarray,
        output_path: str | Path,
        *,
        title: str | None = None,
) -> None:
    """Save a grid occupancy snapshot image."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    occupancy = (grid != -1).astype(float)

    plt.figure(figsize=(5, 5))
    plt.imshow(occupancy, cmap="viridis", interpolation="nearest", vmin=0.0, vmax=1.0)
    if title:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
