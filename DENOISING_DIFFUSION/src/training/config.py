"""Configuration schema and IO helpers for training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
import copy
import json

import yaml


@dataclass
class ProjectMetadata:
    name: str = "exxa-denoising-diffusion"
    seed: int = 42


@dataclass
class DataConfig:
    train_noisy_path: str = "data/dirty.npy"
    train_clean_path: str = "data/clean.npy"
    image_size: int = 64
    patch_size: int = 64
    patches_per_image: int = 4
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 2


@dataclass
class ModelConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 64
    channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    use_amp: bool = False
    device: str = "auto"


@dataclass
class LoggingConfig:
    log_every_n_steps: int = 100
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 5
    output_dir: str = "experiments"
    use_wandb: bool = True
    wandb_project: str = "exxa-denoising"


@dataclass
class ProjectConfig:
    project: ProjectMetadata = field(default_factory=ProjectMetadata)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.data.image_size <= 0:
            raise ValueError("data.image_size must be > 0")
        if self.training.batch_size <= 0:
            raise ValueError("training.batch_size must be > 0")
        if self.training.learning_rate <= 0:
            raise ValueError("training.learning_rate must be > 0")
        if self.diffusion.timesteps <= 0:
            raise ValueError("diffusion.timesteps must be > 0")
        if self.diffusion.beta_schedule not in {"linear", "cosine"}:
            raise ValueError("diffusion.beta_schedule must be one of: linear, cosine")
        split_total = self.data.val_split + self.data.test_split
        if not (0.0 <= self.data.val_split < 1.0 and 0.0 <= self.data.test_split < 1.0):
            raise ValueError("data.val_split and data.test_split must be in [0, 1)")
        if split_total >= 1.0:
            raise ValueError("data.val_split + data.test_split must be < 1")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectConfig":
        data = copy.deepcopy(payload)
        cfg = cls(
            project=ProjectMetadata(**data.get("project", {})),
            data=DataConfig(**data.get("data", {})),
            model=ModelConfig(**data.get("model", {})),
            diffusion=DiffusionConfig(**data.get("diffusion", {})),
            training=TrainingConfig(**data.get("training", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )
        cfg.validate()
        return cfg


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(f) or {}
        elif suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config extension: {suffix}")
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def load_config(
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ProjectConfig:
    """Load project config from file and optional nested overrides."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "base_config.yaml"
    path = Path(config_path)
    payload = _read_config_file(path)
    if overrides:
        payload = _deep_merge(payload, overrides)
    return ProjectConfig.from_dict(payload)


def save_config(config: ProjectConfig, config_path: str | Path) -> None:
    """Save config to YAML or JSON based on destination extension."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict()

    with path.open("w", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            yaml.safe_dump(data, f, sort_keys=False)
        elif path.suffix.lower() == ".json":
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config extension: {path.suffix}")
