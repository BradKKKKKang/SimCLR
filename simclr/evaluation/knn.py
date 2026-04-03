from __future__ import annotations

import torch
from tqdm import tqdm


@torch.no_grad()
def knn_evaluate(
    model,
    memory_data_loader,
    test_data_loader,
    *,
    knn_k: int,
    temperature: float,
    num_classes: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    feature_bank = []

    for data, _ in tqdm(memory_data_loader, desc="kNN feature bank", leave=False):
        feature, _ = model(data.to(device, non_blocking=True))
        feature_bank.append(feature)

    feature_bank_tensor = torch.cat(feature_bank, dim=0).t().contiguous()
    feature_labels = torch.tensor(
        memory_data_loader.dataset.get_targets(),
        device=device,
        dtype=torch.long,
    )

    total_top1 = 0.0
    total_top5 = 0.0
    total_num = 0
    effective_k = min(knn_k, feature_bank_tensor.size(1))
    topk_limit = min(5, num_classes)

    for data, target in tqdm(test_data_loader, desc="kNN eval", leave=False):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        feature, _ = model(data)

        total_num += data.size(0)
        sim_matrix = torch.mm(feature, feature_bank_tensor)
        sim_weight, sim_indices = sim_matrix.topk(k=effective_k, dim=-1)
        sim_labels = torch.gather(
            feature_labels.expand(data.size(0), -1),
            dim=-1,
            index=sim_indices,
        )
        sim_weight = (sim_weight / temperature).exp()

        one_hot_label = torch.zeros(data.size(0) * effective_k, num_classes, device=device)
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        pred_scores = torch.sum(
            one_hot_label.view(data.size(0), -1, num_classes) * sim_weight.unsqueeze(dim=-1),
            dim=1,
        )

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        total_top1 += torch.sum(
            (pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()
        ).item()
        total_top5 += torch.sum(
            (pred_labels[:, :topk_limit] == target.unsqueeze(dim=-1)).any(dim=-1).float()
        ).item()

    return {
        "knn_acc@1": total_top1 / total_num * 100.0,
        "knn_acc@5": total_top5 / total_num * 100.0,
    }
