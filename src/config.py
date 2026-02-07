"""Configuration helpers for the congestion simulation."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Config:
    """Simulation configuration.

    Extension hooks (not implemented): turning probability (behavior),
    variable speed, road masks/intersections, traffic lights.
    """

    N: int = 50
    densities: list[float] = field(
        default_factory=lambda: [
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
        ]
    )
    burn_in_steps: int = 200
    measurement_steps: int = 1000
    replications: int = 10
    seed_base: int = 123
    p_turn: float = 0.0
    output_dir: str = "results"
    save_snapshots: bool = False
    snapshot_densities: list[float] = field(default_factory=lambda: [0.1, 0.4, 0.7])
    snapshot_step: str = "last"
    save_time_series: bool = False
    time_series_densities: list[float] = field(default_factory=list)
    time_series_replication: int = 0


def load_config (path: str | Path) -> Config:
    """Load configuration from a JSON file, overriding defaults."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = Config()
    _apply_dict(cfg, data)
    return cfg


def save_config (path: str | Path, config: Config) -> None:
    """Save configuration to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)


def _apply_dict (config: Config, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if not hasattr(config, key):
            continue
        if key == "densities":
            setattr(config, key, _coerce_floats(value))
        elif key == "snapshot_densities":
            setattr(config, key, _coerce_floats(value))
        elif key == "time_series_densities":
            setattr(config, key, _coerce_floats(value))
        elif key == "p_turn":
            setattr(config, key, float(value))
        else:
            setattr(config, key, value)


def _coerce_floats (values: Iterable[Any]) -> list[float]:
    return [float(v) for v in values]
