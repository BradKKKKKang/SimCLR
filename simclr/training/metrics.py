from __future__ import annotations

import math
from collections import defaultdict

import torch


class MeanMetricTracker:
    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def update(self, metrics: dict[str, float], n: int) -> None:
        for key, value in metrics.items():
            self._sums[key] += float(value) * n
            self._counts[key] += n

    def compute(self, prefix: str = "") -> dict[str, float]:
        result: dict[str, float] = {}
        for key, total in self._sums.items():
            count = self._counts[key]
            result[f"{prefix}{key}"] = total / count if count else 0.0
        return result


class PositiveCosineAccumulator:
    def __init__(self) -> None:
        self._batches: list[torch.Tensor] = []

    def update(self, batch_pos_cos: torch.Tensor) -> None:
        self._batches.append(batch_pos_cos.detach().cpu())

    def compute(self) -> dict[str, float]:
        if not self._batches:
            return {
                "pos_cos_mean": 0.0,
                "pos_cos_std": 0.0,
                "pos_cos_min": 0.0,
                "pos_cos_tail_mean_10": 0.0,
                "pos_cos_tail_mean_25": 0.0,
            }

        pos_cos = torch.cat(self._batches, dim=0).float()
        sorted_pos_cos, _ = torch.sort(pos_cos)
        return {
            "pos_cos_mean": pos_cos.mean().item(),
            "pos_cos_std": pos_cos.std(unbiased=False).item(),
            "pos_cos_min": pos_cos.min().item(),
            "pos_cos_tail_mean_10": _compute_tail_mean(sorted_pos_cos, 0.10),
            "pos_cos_tail_mean_25": _compute_tail_mean(sorted_pos_cos, 0.25),
        }


def _compute_tail_mean(sorted_values: torch.Tensor, tail_ratio: float) -> float:
    tail_count = max(1, math.ceil(sorted_values.numel() * tail_ratio))
    return sorted_values[:tail_count].mean().item()
