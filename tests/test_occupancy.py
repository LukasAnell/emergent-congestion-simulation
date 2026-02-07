import numpy as np

from src.simulation import initialize_state
from src.utils import make_rng


def test_occupancy_invariant():
    rng = make_rng(42)
    state = initialize_state(N=10, density=0.3, rng=rng)
    A = int(state.x.size)

    for _ in range(50):
        state.step()
        occupied = state.grid != -1
        assert int(np.count_nonzero(occupied)) == A
        ids = state.grid[occupied]
        assert np.unique(ids).size == A
