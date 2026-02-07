import numpy as np

from src.config import Config
from src.simulation import SimulationState, initialize_state, run_simulation


def test_initialize_state_zero_agents():
    rng = np.random.default_rng(1)
    state = initialize_state(N=5, density=0.0, rng=rng)

    assert state.x.size == 0
    assert state.y.size == 0
    assert state.d.size == 0
    assert np.all(state.grid == -1)


def test_step_no_allowed_moves_due_to_occupancy():
    N = 2
    grid = np.full((N, N), -1, dtype=int)
    x = np.array([0, 1], dtype=int)
    y = np.array([0, 0], dtype=int)
    d = np.array([1, 3], dtype=int)  # East and West, targets occupied at time t
    grid[y, x] = [0, 1]

    state = SimulationState(N=N, grid=grid, x=x, y=y, d=d)
    moved = state.step()

    assert moved == 0
    assert state.x.tolist() == [0, 1]
    assert state.y.tolist() == [0, 0]


def test_step_with_no_agents_returns_zero():
    N = 3
    grid = np.full((N, N), -1, dtype=int)
    state = SimulationState(
        N=N,
        grid=grid,
        x=np.array([], dtype=int),
        y=np.array([], dtype=int),
        d=np.array([], dtype=int),
    )

    moved = state.step()

    assert moved == 0
    assert np.all(state.grid == -1)


def test_run_simulation_with_zero_measurement_steps():
    cfg = Config(N=5, burn_in_steps=1, measurement_steps=0, replications=1)
    metrics = run_simulation(cfg, seed=1, density=0.2)

    assert metrics["mean_v"] == 0.0
    assert metrics["mean_b"] == 0.0
    assert metrics["A"] == int(np.floor(0.2 * 5 * 5))
