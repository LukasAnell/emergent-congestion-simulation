def test_wraparound_east (state_factory):
    N = 3
    state = state_factory(N, positions=[(2, 1)], directions=[1])  # East
    moved = state.step()

    assert moved == 1
    assert state.x[0] == 0
    assert state.y[0] == 1
    assert state.grid[1, 0] == 0


def test_wraparound_north (state_factory):
    N = 3
    state = state_factory(N, positions=[(1, 0)], directions=[0])  # North
    moved = state.step()

    assert moved == 1
    assert state.x[0] == 1
    assert state.y[0] == 2
    assert state.grid[2, 1] == 0
