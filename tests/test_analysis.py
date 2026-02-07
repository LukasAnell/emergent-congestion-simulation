from src.analysis import estimate_critical_density


def test_estimate_critical_density_steepest_drop_segment ():
    densities = [0.1, 0.2, 0.3, 0.4]
    mean_speeds = [0.9, 0.85, 0.4, 0.35]

    critical_density_est, critical_drop_est = estimate_critical_density(densities, mean_speeds)

    assert critical_density_est == 0.25
    assert critical_drop_est == -0.45


def test_estimate_critical_density_single_point ():
    critical_density_est, critical_drop_est = estimate_critical_density([0.2], [0.5])

    assert critical_density_est == 0.2
    assert critical_drop_est == 0.0
