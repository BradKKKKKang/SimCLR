from __future__ import annotations

import csv
import importlib.util
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from simclr.config.schema import TrainConfig, default_run_name, save_config


def prepare_run_directory(config: TrainConfig) -> Path:
    run_name = config.logging.run_name or default_run_name(config)
    output_root = Path(config.logging.output_dir)
    run_dir = output_root / run_name

    if run_dir.exists() and any(run_dir.iterdir()):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / f"{run_name}_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_metrics_csv(history: list[dict[str, Any]], csv_path: str | Path) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    keys = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def save_run_config(config: TrainConfig, run_dir: str | Path) -> None:
    save_config(config, Path(run_dir) / "config.yaml")


def save_checkpoint(
    checkpoint_path: str | Path,
    *,
    model,
    epoch: int,
    metrics: dict[str, Any],
    config: TrainConfig,
) -> None:
    payload = {
        "epoch": epoch,
        "metrics": metrics,
        "config": asdict(config),
        "model_state": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)


class WandbLogger:
    def __init__(self, *, enabled: bool, config: TrainConfig, run_dir: str | Path):
        self.enabled = enabled
        self._wandb = None
        self._run = None

        if not enabled:
            return

        if importlib.util.find_spec("wandb") is None:
            raise ImportError(
                "wandb is not installed. Install it or set logging.use_wandb=false in the config."
            )

        import wandb

        project = os.environ.get("WANDB_PROJECT")
        if not project:
            raise RuntimeError("WANDB_PROJECT must be set when logging.use_wandb=true.")

        init_kwargs: dict[str, Any] = {
            "project": project,
            "name": config.logging.run_name or default_run_name(config),
            "mode": os.environ.get("WANDB_MODE", "online"),
            "config": asdict(config),
            "dir": os.environ.get("WANDB_DIR", str(run_dir)),
        }
        entity = os.environ.get("WANDB_ENTITY")
        if entity:
            init_kwargs["entity"] = entity

        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key

        self._wandb = wandb
        self._run = wandb.init(**init_kwargs)

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        if self._wandb is None:
            return
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
