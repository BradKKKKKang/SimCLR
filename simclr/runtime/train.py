from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from simclr.config.schema import TrainConfig, default_run_name
from simclr.data.datasets import build_pretrain_datasets, get_dataset_metadata
from simclr.models.simclr_model import Model
from simclr.runtime.tracking import (
    WandbLogger,
    prepare_run_directory,
    save_checkpoint,
    save_metrics_csv,
    save_run_config,
)
from simclr.training.losses import CompositeContrastiveLoss, CompositeContrastiveLossConfig
from simclr.training.trainer import fit


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(config: TrainConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    return optim.Adam(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )


def build_scheduler(
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
) -> LRScheduler:
    warmup_epochs = config.optimizer.warmup_epochs

    def lr_lambda(epoch_idx: int) -> float:
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        return 1.0

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_dataloaders(config: TrainConfig):
    metadata = get_dataset_metadata(config.dataset.name)
    train_data, memory_data, test_data = build_pretrain_datasets(
        config.dataset.name,
        config.dataset.data_root,
        download=config.dataset.download,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_data,
        batch_size=config.runtime.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    eval_batch_size = config.runtime.batch_size
    memory_loader = DataLoader(
        memory_data,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=pin_memory,
    )
    return metadata, train_loader, memory_loader, test_loader


def build_loss(config: TrainConfig) -> CompositeContrastiveLoss:
    return CompositeContrastiveLoss(
        CompositeContrastiveLossConfig(
            temperature=config.loss.temperature,
            hard_ratio=config.loss.hard_ratio,
            reg_form=config.loss.reg_form,
            reg_weight=config.loss.reg_weight,
        )
    )


def run_training(config: TrainConfig) -> Path:
    config.validate()
    if config.logging.run_name is None:
        config.logging.run_name = default_run_name(config)

    device = select_device(config.runtime.device)
    set_seed(config.runtime.seed)
    run_dir = prepare_run_directory(config)
    save_run_config(config, run_dir)

    metadata, train_loader, memory_loader, test_loader = build_dataloaders(config)
    model = Model(config.model.feature_dim).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    criterion = build_loss(config)
    wandb_logger = WandbLogger(enabled=config.logging.use_wandb, config=config, run_dir=run_dir)
    history: list[dict[str, float | int | str]] = []

    def handle_epoch_end(epoch_metrics: dict[str, float | int | str], is_best: bool) -> None:
        history.append(epoch_metrics)
        save_metrics_csv(history, run_dir / "metrics.csv")
        wandb_logger.log(epoch_metrics, step=int(epoch_metrics["epoch"]))
        if is_best:
            save_checkpoint(
                run_dir / "best_knn.pth",
                model=model,
                epoch=int(epoch_metrics["epoch"]),
                metrics=epoch_metrics,
                config=config,
            )

    try:
        fit(
            model=model,
            train_loader=train_loader,
            memory_loader=memory_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            total_epochs=config.runtime.epochs,
            knn_k=config.eval.knn_k,
            knn_temperature=config.loss.temperature,
            num_classes=metadata.num_classes,
            on_epoch_end=handle_epoch_end,
        )
    finally:
        last_metrics = history[-1] if history else {}
        last_epoch = int(last_metrics.get("epoch", config.runtime.epochs))
        save_checkpoint(
            run_dir / "last.pth",
            model=model,
            epoch=last_epoch,
            metrics=last_metrics,
            config=config,
        )
        wandb_logger.finish()

    return run_dir
