"""Core simulation logic for congestion on a periodic grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .utils import make_rng

# Direction codes: 0=N, 1=E, 2=S, 3=W
DIRECTION_OFFSETS = {
    0: (0, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0),
}


@dataclass
class SimulationState:
    """Holds grid and agent state.

    Note: grid is indexed as grid[y, x] where x is column and y is row.
    """

    N: int
    grid: np.ndarray
    x: np.ndarray
    y: np.ndarray
    d: np.ndarray

    def step(
            self,
            *,
            rng: np.random.Generator | None = None,
            p_turn: float = 0.0,
    ) -> int:
        """Advance one timestep with synchronous updates.

        Returns the number of agents that moved. When p_turn > 0, agents turn
        left or right with probability p_turn before proposing moves.
        """
        if self.x.size == 0:
            return 0

        if p_turn > 0.0:
            if rng is None:
                rng = np.random.default_rng()
            if p_turn >= 1.0:
                turn_mask = np.ones(self.d.size, dtype=bool)
            else:
                turn_mask = rng.random(self.d.size) < p_turn
            if np.any(turn_mask):
                turn_delta = rng.choice([1, 3], size=int(np.sum(turn_mask)))
                self.d[turn_mask] = (self.d[turn_mask] + turn_delta) % 4

        dx = np.zeros_like(self.x)
        dy = np.zeros_like(self.y)

        dx[self.d == 1] = 1
        dx[self.d == 3] = -1
        dy[self.d == 0] = -1
        dy[self.d == 2] = 1

        nx = (self.x + dx) % self.N
        ny = (self.y + dy) % self.N

        # Extension hook: variable speed could propose multi-cell moves.
        # Extension hook: road masks/traffic lights could filter allowed moves.

        empty_mask = (self.grid[ny, nx] == -1)
        targets = nx + self.N * ny
        counts = np.bincount(targets, minlength=self.N * self.N)
        allowed = empty_mask & (counts[targets] == 1)

        if not np.any(allowed):
            return 0

        movers = np.where(allowed)[0]
        self.grid[self.y[movers], self.x[movers]] = -1
        self.x[movers] = nx[movers]
        self.y[movers] = ny[movers]
        self.grid[self.y[movers], self.x[movers]] = movers

        return int(movers.size)


def initialize_state(N: int, density: float, rng: np.random.Generator) -> SimulationState:
    """Initialize grid and agents for a given density."""
    A = int(np.floor(density * N * N))
    grid = np.full((N, N), -1, dtype=int)
    if A == 0:
        return SimulationState(
            N=N, grid=grid, x=np.array([], dtype=int), y=np.array([], dtype=int), d=np.array([], dtype=int)
        )

    cells = rng.choice(N * N, size=A, replace=False)
    y = cells // N
    x = cells % N
    d = rng.integers(0, 4, size=A, dtype=int)

    grid[y, x] = np.arange(A, dtype=int)

    return SimulationState(N=N, grid=grid, x=x, y=y, d=d)


def run_simulation(
        config: Config,
        seed: int,
        density: float,
        snapshot_step: str | int | None = None,
        collect_time_series: bool = False,
) -> dict[str, float | int | np.ndarray | list[dict[str, float | int]]]:
    """Run a single simulation replication and return mean metrics."""
    rng = make_rng(seed)
    state = initialize_state(config.N, density, rng)
    A = int(state.x.size)

    snapshot_idx = None
    if snapshot_step is not None:
        snapshot_idx = _resolve_snapshot_index(snapshot_step, int(config.measurement_steps))
    snapshot_grid: np.ndarray | None = None
    time_series: list[dict[str, float | int]] = []

    for _ in range(int(config.burn_in_steps)):
        state.step(rng=rng, p_turn=config.p_turn)

    if config.measurement_steps <= 0:
        if snapshot_idx == 0:
            snapshot_grid = state.grid.copy()
        result: dict[str, float | int | np.ndarray | list[dict[str, float | int]]] = {
            "mean_v": 0.0,
            "mean_b": 0.0,
            "A": A,
        }
        if snapshot_grid is not None:
            result["snapshot_grid"] = snapshot_grid
        if collect_time_series:
            result["time_series"] = time_series
        return result

    sum_v = 0.0
    sum_b = 0.0
    for step_idx in range(int(config.measurement_steps)):
        moved = state.step(rng=rng, p_turn=config.p_turn)
        v = (moved / A) if A > 0 else 0.0
        blocked = 1.0 - v
        sum_v += v
        sum_b += blocked
        if snapshot_idx is not None and step_idx == snapshot_idx:
            snapshot_grid = state.grid.copy()
        if collect_time_series:
            time_series.append(
                {
                    "timestep": int(step_idx),
                    "moved_count": int(moved),
                    "speed": float(v),
                    "blocked_fraction": float(blocked),
                }
            )

    mean_v = sum_v / float(config.measurement_steps)
    mean_b = sum_b / float(config.measurement_steps)

    result = {"mean_v": mean_v, "mean_b": mean_b, "A": A}
    if snapshot_grid is not None:
        result["snapshot_grid"] = snapshot_grid
    if collect_time_series:
        result["time_series"] = time_series
    return result


def _resolve_snapshot_index(snapshot_step: str | int, measurement_steps: int) -> int | None:
    if measurement_steps <= 0:
        return 0
    if snapshot_step == "last":
        return measurement_steps - 1
    if isinstance(snapshot_step, int):
        if snapshot_step < 0 or snapshot_step >= measurement_steps:
            raise ValueError(
                "snapshot_step as int must satisfy 0 <= snapshot_step < measurement_steps"
            )
        return snapshot_step
    if isinstance(snapshot_step, str) and snapshot_step.strip().isdigit():
        numeric_step = int(snapshot_step.strip())
        if numeric_step < 0 or numeric_step >= measurement_steps:
            raise ValueError(
                "snapshot_step as int must satisfy 0 <= snapshot_step < measurement_steps"
            )
        return numeric_step
    raise ValueError("snapshot_step must be 'last' or an integer index")
