import numpy as np

from src.simulation import SimulationState


def test_turning_probability_changes_direction():
    N = 4
    grid = np.full((N, N), -1, dtype=int)
    x = np.array([0, 1, 2], dtype=int)
    y = np.array([0, 1, 2], dtype=int)
    d = np.array([0, 1, 2], dtype=int)
    grid[y, x] = np.arange(x.size)

    state = SimulationState(N=N, grid=grid, x=x, y=y, d=d)
    rng = np.random.default_rng(0)

    old_d = state.d.copy()
    state.step(rng=rng, p_turn=1.0)

    assert np.all(state.d != old_d)
    delta = (state.d - old_d) % 4
    assert np.all(np.isin(delta, [1, 3]))
