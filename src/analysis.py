"""Post-processing helpers for experiment outputs."""
from __future__ import annotations

from typing import Sequence


def estimate_critical_density(
        densities: Sequence[float],
        mean_speeds: Sequence[float],
) -> tuple[float, float]:
    """Estimate critical density from steepest speed drop.

    Returns (critical_density_est, critical_drop_est).
    """
    if len(densities) != len(mean_speeds):
        raise ValueError("densities and mean_speeds must have the same length")
    if len(densities) == 0:
        raise ValueError("at least one density is required")
    if len(densities) == 1:
        return float(densities[0]), 0.0

    min_slope = None
    min_idx = 0

    for idx in range(len(densities) - 1):
        rho0 = float(densities[idx])
        rho1 = float(densities[idx + 1])
        if rho1 <= rho0:
            raise ValueError("densities must be strictly increasing")

        v0 = float(mean_speeds[idx])
        v1 = float(mean_speeds[idx + 1])
        slope = (v1 - v0) / (rho1 - rho0)

        if min_slope is None or slope < min_slope:
            min_slope = slope
            min_idx = idx

    critical_density_est = 0.5 * (float(densities[min_idx]) + float(densities[min_idx + 1]))
    critical_drop_est = float(mean_speeds[min_idx + 1]) - float(mean_speeds[min_idx])
    return critical_density_est, critical_drop_est
