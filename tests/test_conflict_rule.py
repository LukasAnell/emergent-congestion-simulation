import numpy as np

from src.simulation import SimulationState


def test_conflict_rule_blocks_all ():
    N = 3
    grid = np.full((N, N), -1, dtype=int)
    x = np.array([0, 2], dtype=int)
    y = np.array([1, 1], dtype=int)
    d = np.array([1, 3], dtype=int)  # East and West, both target (1,1)
    grid[y, x] = [0, 1]

    state = SimulationState(N=N, grid=grid, x=x, y=y, d=d)
    moved = state.step()

    assert moved == 0
    assert state.x.tolist() == [0, 2]
    assert state.y.tolist() == [1, 1]
    assert state.grid[1, 0] == 0
    assert state.grid[1, 2] == 1


def test_blocked_by_occupancy ():
    N = 3
    grid = np.full((N, N), -1, dtype=int)
    x = np.array([0, 1], dtype=int)
    y = np.array([0, 0], dtype=int)
    d = np.array([1, 1], dtype=int)  # Both East
    grid[y, x] = [0, 1]

    state = SimulationState(N=N, grid=grid, x=x, y=y, d=d)
    moved = state.step()

    assert moved == 1
    assert state.x[0] == 0  # blocked by occupancy at time t
    assert state.y[0] == 0
    assert state.x[1] == 2  # moved into empty cell
    assert state.y[1] == 0
