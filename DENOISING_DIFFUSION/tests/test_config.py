"""Unit tests for training configuration schema and IO."""

from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.config import ProjectConfig, load_config, save_config


def test_load_default_config() -> None:
    cfg = load_config()
    assert isinstance(cfg, ProjectConfig)
    assert cfg.project.name == "exxa-denoising-diffusion"
    assert cfg.diffusion.timesteps == 1000


def test_load_with_overrides() -> None:
    cfg = load_config(
        overrides={
            "training": {"batch_size": 8},
            "diffusion": {"beta_schedule": "cosine"},
        }
    )
    assert cfg.training.batch_size == 8
    assert cfg.diffusion.beta_schedule == "cosine"


def test_load_json_config(tmp_path: Path) -> None:
    config_path = tmp_path / "custom_config.json"
    payload = {
        "project": {"name": "json-test", "seed": 7},
        "data": {"image_size": 128},
        "model": {"base_channels": 32},
        "diffusion": {"timesteps": 200, "beta_schedule": "linear"},
        "training": {"batch_size": 4, "learning_rate": 1e-4},
        "logging": {"use_wandb": False},
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = load_config(config_path)
    assert cfg.project.name == "json-test"
    assert cfg.data.image_size == 128
    assert cfg.training.batch_size == 4


def test_save_and_reload_yaml(tmp_path: Path) -> None:
    cfg = load_config(overrides={"project": {"name": "saved-test"}})
    out_path = tmp_path / "saved_config.yaml"

    save_config(cfg, out_path)
    loaded = load_config(out_path)

    assert loaded.project.name == "saved-test"
    assert loaded.model.base_channels == cfg.model.base_channels


def test_validation_rejects_bad_split() -> None:
    with pytest.raises(ValueError, match=r"val_split \+ data.test_split"):
        load_config(overrides={"data": {"val_split": 0.7, "test_split": 0.4}})


def test_validation_rejects_bad_schedule() -> None:
    with pytest.raises(ValueError, match="beta_schedule"):
        load_config(overrides={"diffusion": {"beta_schedule": "invalid"}})
