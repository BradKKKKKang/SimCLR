from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from tqdm import tqdm

from simclr.evaluation.knn import knn_evaluate
from simclr.training.losses import CompositeContrastiveLoss
from simclr.training.metrics import MeanMetricTracker, PositiveCosineAccumulator


def train_one_epoch(
    *,
    model: torch.nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    criterion: CompositeContrastiveLoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    model.train()
    loss_tracker = MeanMetricTracker()
    pos_cos_accumulator = PositiveCosineAccumulator()
    train_bar = tqdm(data_loader, desc=f"train {epoch}/{total_epochs}", leave=False)

    for pos_1, pos_2, _ in train_bar:
        pos_1 = pos_1.to(device, non_blocking=True)
        pos_2 = pos_2.to(device, non_blocking=True)
        _, proj_1 = model(pos_1)
        _, proj_2 = model(pos_2)

        total_loss, loss_metrics, batch_pos_cos = criterion(proj_1, proj_2)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        batch_size = pos_1.size(0)
        loss_tracker.update(
            {
                "loss": total_loss.detach().item(),
                "infonce_loss": loss_metrics["infonce_loss"].item(),
                "reg_loss": loss_metrics["reg_loss"].item(),
            },
            n=batch_size,
        )
        pos_cos_accumulator.update(batch_pos_cos)

        running_metrics = loss_tracker.compute(prefix="train_")
        train_bar.set_postfix(
            loss=f"{running_metrics['train_loss']:.4f}",
            reg=f"{running_metrics['train_reg_loss']:.4f}",
        )

    epoch_metrics = loss_tracker.compute(prefix="train_")
    epoch_metrics.update(pos_cos_accumulator.compute())
    return epoch_metrics


def fit(
    *,
    model: torch.nn.Module,
    train_loader,
    memory_loader,
    test_loader,
    optimizer: torch.optim.Optimizer,
    criterion: CompositeContrastiveLoss,
    device: torch.device,
    total_epochs: int,
    knn_k: int,
    knn_temperature: float,
    num_classes: int,
    on_epoch_end: Callable[[dict[str, Any], bool], None] | None = None,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    best_knn_acc1 = float("-inf")

    for epoch in range(1, total_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        knn_metrics = knn_evaluate(
            model,
            memory_loader,
            test_loader,
            knn_k=knn_k,
            temperature=knn_temperature,
            num_classes=num_classes,
            device=device,
        )

        epoch_metrics: dict[str, Any] = {"epoch": epoch}
        epoch_metrics.update(train_metrics)
        epoch_metrics.update(knn_metrics)
        history.append(epoch_metrics)

        is_best = knn_metrics["knn_acc@1"] > best_knn_acc1
        if is_best:
            best_knn_acc1 = knn_metrics["knn_acc@1"]

        if on_epoch_end is not None:
            on_epoch_end(epoch_metrics, is_best)

    return history
