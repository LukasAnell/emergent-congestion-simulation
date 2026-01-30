import uuid
from pathlib import Path

import pytest
import numpy as np

from src.simulation import SimulationState


@pytest.fixture()
def local_tmp_path ():
    base = Path.cwd() / "tests" / "tmp_work"
    base.mkdir(parents=True, exist_ok=True)
    tmp = base / uuid.uuid4().hex
    tmp.mkdir(parents=True, exist_ok=True)
    yield tmp


@pytest.fixture()
def state_factory ():
    """Create small deterministic SimulationState fixtures for micro-scenarios."""

    def _make_state (N: int, positions: list[tuple[int, int]], directions: list[int]) -> SimulationState:
        if len(positions) != len(directions):
            raise ValueError("positions and directions must be the same length")
        grid = np.full((N, N), -1, dtype=int)
        if positions:
            x = np.array([p[0] for p in positions], dtype=int)
            y = np.array([p[1] for p in positions], dtype=int)
            d = np.array(directions, dtype=int)
            grid[y, x] = np.arange(len(positions), dtype=int)
        else:
            x = np.array([], dtype=int)
            y = np.array([], dtype=int)
            d = np.array([], dtype=int)
        return SimulationState(N=N, grid=grid, x=x, y=y, d=d)

    return _make_state
