import json

from src.config import Config, load_config, save_config


def test_save_and_load_config_roundtrip(local_tmp_path):
    cfg = Config()
    cfg.N = 12
    cfg.densities = [0.2, 0.4]
    cfg.snapshot_densities = [0.3]
    cfg.output_dir = "results-test"

    path = local_tmp_path / "config.json"
    save_config(path, cfg)

    loaded = load_config(path)
    assert loaded.N == 12
    assert loaded.densities == [0.2, 0.4]
    assert loaded.snapshot_densities == [0.3]
    assert loaded.output_dir == "results-test"


def test_load_config_ignores_unknown_keys(local_tmp_path):
    data = {
        "N": 7,
        "densities": [0.1, 0.2],
        "snapshot_densities": [0.5],
        "unknown_key": 123,
    }
    path = local_tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_config(path)
    assert loaded.N == 7
    assert loaded.densities == [0.1, 0.2]
    assert loaded.snapshot_densities == [0.5]
    assert not hasattr(loaded, "unknown_key")
