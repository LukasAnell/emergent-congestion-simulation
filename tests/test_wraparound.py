import numpy as np

from src.simulation import SimulationState


def test_wraparound_east ():
    N = 3
    grid = np.full((N, N), -1, dtype=int)
    x = np.array([2], dtype=int)
    y = np.array([1], dtype=int)
    d = np.array([1], dtype=int)  # East
    grid[y, x] = 0

    state = SimulationState(N=N, grid=grid, x=x, y=y, d=d)
    moved = state.step()

    assert moved == 1
    assert state.x[0] == 0
    assert state.y[0] == 1
    assert state.grid[1, 0] == 0
