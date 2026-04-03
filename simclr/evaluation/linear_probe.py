from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from simclr.data.datasets import build_linear_eval_datasets, get_dataset_metadata
from simclr.models.simclr_model import Model, load_checkpoint_file, load_pretrained_encoder


def save_results_csv(results, csv_path):
    keys = list(results.keys())
    row_count = len(results[keys[0]]) if keys else 0
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch"] + keys)
        for index in range(row_count):
            writer.writerow([index + 1] + [results[key][index] for key in keys])


class LinearProbeNet(nn.Module):
    def __init__(self, num_class: int, feature_dim: int, pretrained_path: str):
        super().__init__()
        backbone = Model(feature_dim=feature_dim)
        load_pretrained_encoder(backbone, pretrained_path)
        self.f = backbone.f
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        return self.fc(feature)


def train_or_eval_epoch(net, data_loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
    bar = tqdm(data_loader, leave=False)
    with torch.enable_grad() if is_train else torch.no_grad():
        for data, target in bar:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            out = net(data)
            loss = criterion(out, target)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, : min(5, out.size(1))] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            bar.set_postfix(
                loss=f"{total_loss / total_num:.4f}",
                acc1=f"{total_correct_1 / total_num * 100:.2f}",
            )

    return (
        total_loss / total_num,
        total_correct_1 / total_num * 100,
        total_correct_5 / total_num * 100,
    )


def run_linear_probe(
    *,
    model_path: str,
    dataset: str = "cifar10",
    data_root: str = "data",
    batch_size: int = 512,
    epochs: int = 100,
    num_workers: int = 4,
    device_name: str = "auto",
    output_dir: str = "linear_probe_runs",
    download: bool = True,
) -> tuple[Path, Path]:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    metadata = get_dataset_metadata(dataset)
    train_data, test_data = build_linear_eval_datasets(dataset, data_root, download=download)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = load_checkpoint_file(model_path)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        feature_dim = checkpoint["config"]["model"]["feature_dim"]
    else:
        feature_dim = 128

    model = LinearProbeNet(
        num_class=metadata.num_classes,
        feature_dim=feature_dim,
        pretrained_path=model_path,
    ).to(device)
    for param in model.f.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / f"{Path(model_path).stem}_{dataset}_linear_probe.csv"

    results = {
        "train_loss": [],
        "train_acc@1": [],
        "train_acc@5": [],
        "test_loss": [],
        "test_acc@1": [],
        "test_acc@5": [],
    }

    best_acc = float("-inf")
    best_model_path = output_dir_path / f"{Path(model_path).stem}_{dataset}_linear_probe_best.pth"

    for _epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_or_eval_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_acc_1, test_acc_5 = train_or_eval_epoch(model, test_loader, criterion, device)

        results["train_loss"].append(train_loss)
        results["train_acc@1"].append(train_acc_1)
        results["train_acc@5"].append(train_acc_5)
        results["test_loss"].append(test_loss)
        results["test_acc@1"].append(test_acc_1)
        results["test_acc@5"].append(test_acc_5)
        save_results_csv(results, output_path)

        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), best_model_path)

    return output_path, best_model_path
