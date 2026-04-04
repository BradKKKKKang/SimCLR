from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    data_root: str = "data"
    num_workers: int = 4
    download: bool = True


@dataclass
class ModelConfig:
    feature_dim: int = 128


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 5e-4
    weight_decay: float = 1e-6
    warmup_epochs: int = 5


@dataclass
class RuntimeConfig:
    batch_size: int = 512
    epochs: int = 500
    seed: int = 42
    device: str = "auto"


@dataclass
class LossConfig:
    temperature: float = 0.5
    hard_ratio: float = 0.25
    reg_form: str = "square"
    reg_weight: float = 0.0


@dataclass
class EvalConfig:
    knn_k: int = 200


@dataclass
class LoggingConfig:
    output_dir: str = "runs"
    use_wandb: bool = False
    run_name: str | None = None


@dataclass
class TrainConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        if self.runtime.batch_size <= 0:
            raise ValueError("runtime.batch_size must be positive.")
        if self.runtime.epochs <= 0:
            raise ValueError("runtime.epochs must be positive.")
        if self.loss.temperature <= 0:
            raise ValueError("loss.temperature must be positive.")
        if not 0.0 < self.loss.hard_ratio <= 1.0:
            raise ValueError("loss.hard_ratio must be in the range (0, 1].")
        if self.loss.reg_form not in {"linear", "square"}:
            raise ValueError("loss.reg_form must be one of: linear, square.")
        if self.loss.reg_weight < 0:
            raise ValueError("loss.reg_weight must be non-negative.")
        if self.optimizer.name.lower() != "adam":
            raise ValueError("Only the Adam optimizer is supported in this minimal framework.")
        if self.optimizer.lr <= 0:
            raise ValueError("optimizer.lr must be positive.")
        if self.optimizer.warmup_epochs < 0:
            raise ValueError("optimizer.warmup_epochs must be non-negative.")
        if self.optimizer.warmup_epochs > self.runtime.epochs:
            raise ValueError("optimizer.warmup_epochs must not exceed runtime.epochs.")
        if self.eval.knn_k <= 0:
            raise ValueError("eval.knn_k must be positive.")


SECTION_TYPES = {
    "dataset": DatasetConfig,
    "model": ModelConfig,
    "optimizer": OptimizerConfig,
    "runtime": RuntimeConfig,
    "loss": LossConfig,
    "eval": EvalConfig,
    "logging": LoggingConfig,
}


def _convert_dataclass(cls: type[Any], payload: dict[str, Any]) -> Any:
    values: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name not in payload:
            continue

        value = payload[field_info.name]
        field_type = SECTION_TYPES.get(field_info.name)
        if isinstance(field_type, type) and is_dataclass(field_type) and isinstance(value, dict):
            values[field_info.name] = _convert_dataclass(field_type, value)
        else:
            values[field_info.name] = value
    return cls(**values)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    return asdict(config)


def train_config_from_dict(payload: dict[str, Any]) -> TrainConfig:
    config = _convert_dataclass(TrainConfig, payload)
    config.validate()
    return config


def load_config(config_path: str | Path) -> TrainConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    return train_config_from_dict(raw_config)


def save_config(config: TrainConfig, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)


def default_run_name(config: TrainConfig) -> str:
    reg_mode = "baseline" if config.loss.reg_weight == 0 else (
        "all_positive" if config.loss.hard_ratio == 1.0 else "hard_positive"
    )
    hard_ratio = format(config.loss.hard_ratio, ".4g").replace(".", "p")
    reg_weight = format(config.loss.reg_weight, ".4g").replace(".", "p")
    temperature = format(config.loss.temperature, ".4g").replace(".", "p")
    return (
        f"{config.dataset.name}_"
        f"{reg_mode}_"
        f"bs{config.runtime.batch_size}_"
        f"t{temperature}_"
        f"hr{hard_ratio}_"
        f"{config.loss.reg_form}_"
        f"rw{reg_weight}"
    )


def nested_update(payload: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    cursor = payload
    segments = dotted_key.split(".")
    for segment in segments[:-1]:
        next_value = cursor.get(segment)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[segment] = next_value
        cursor = next_value
    cursor[segments[-1]] = value
    return payload
