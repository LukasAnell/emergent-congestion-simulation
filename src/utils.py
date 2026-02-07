"""Utility helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return its Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_seed(seed_base: int, density: float, rep_index: int) -> int:
    """Deterministic seed from density and replication index.

    Uses a stable formula to avoid Python's randomized hash().
    """
    density_key = int(round(float(density) * 1000))
    return int(seed_base + density_key * 10000 + int(rep_index))


def seed_from_density_rep(seed_base: int, density: float, rep: int) -> int:
    """Backward-compatible wrapper for make_seed()."""
    return make_seed(seed_base, density, rep)


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def parse_density_list(text: str) -> list[float]:
    """Parse comma-separated density list from CLI."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [float(p) for p in parts]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.mean(values))


def std(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.std(values))
