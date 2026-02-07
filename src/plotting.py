"""Plotting helpers for experiment summaries."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def plot_summary(summary_csv_path: str | Path, output_dir: str | Path) -> None:
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


def _plot_errorbar(
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


def plot_snapshot(
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


def plot_turn_sweep_heatmap(
        speed_matrix_csv_path: str | Path,
        output_path: str | Path,
) -> None:
    """Plot mean-speed heatmap for p_turn x density sweep."""
    matrix_df = pd.read_csv(speed_matrix_csv_path, index_col=0)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    x_labels = [f"{float(v):.2f}" for v in matrix_df.columns]
    y_labels = [f"{float(v):.2f}" for v in matrix_df.index]

    plt.figure(figsize=(8, 4.5))
    image = plt.imshow(
        matrix_df.to_numpy(dtype=float),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(image, label="Mean speed")
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(y_labels)), y_labels)
    plt.xlabel("Density")
    plt.ylabel("Turning probability (p_turn)")
    plt.title("Mean Speed Heatmap: Density vs Turning")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
