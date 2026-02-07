from pandas.testing import assert_frame_equal

from src.config import Config
from src.experiment import run_density_sweep


def test_deterministic_sweep_outputs(local_tmp_path):
    cfg = Config(
        N=6,
        densities=[0.2],
        burn_in_steps=0,
        measurement_steps=2,
        replications=2,
        seed_base=123,
        output_dir=str(local_tmp_path / "results"),
    )

    df1 = run_density_sweep(cfg)
    df2 = run_density_sweep(cfg)

    assert_frame_equal(df1, df2, check_exact=True)
