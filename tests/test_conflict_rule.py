def test_conflict_rule_blocks_all (state_factory):
    N = 3
    state = state_factory(
        N,
        positions=[(0, 1), (2, 1)],
        directions=[1, 3],  # East and West, both target (1,1)
    )
    moved = state.step()

    assert moved == 0
    assert state.x.tolist() == [0, 2]
    assert state.y.tolist() == [1, 1]
    assert state.grid[1, 0] == 0
    assert state.grid[1, 2] == 1


def test_blocked_by_occupancy (state_factory):
    N = 3
    state = state_factory(
        N,
        positions=[(0, 0), (1, 0)],
        directions=[1, 1],  # Both East
    )
    moved = state.step()

    assert moved == 1
    assert state.x[0] == 0  # blocked by occupancy at time t
    assert state.y[0] == 0
    assert state.x[1] == 2  # moved into empty cell
    assert state.y[1] == 0
