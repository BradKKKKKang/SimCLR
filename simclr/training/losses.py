from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


def compute_infonce(proj_1: torch.Tensor, proj_2: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = proj_1.size(0)
    representations = torch.cat([proj_1, proj_2], dim=0)
    logits = torch.matmul(representations, representations.t().contiguous()) / temperature
    diagonal_mask = torch.eye(2 * batch_size, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(diagonal_mask, float("-inf"))

    positive_indices = (torch.arange(2 * batch_size, device=logits.device) + batch_size) % (2 * batch_size)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -log_prob[torch.arange(2 * batch_size, device=logits.device), positive_indices].mean()


def compute_positive_regularization(
    proj_1: torch.Tensor,
    proj_2: torch.Tensor,
    hard_ratio: float,
    reg_form: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # We compute cosine once per sample pair. Each batch contributes B unique
    # positive pairs, not 2B, because cosine similarity is symmetric.
    pos_cos = torch.sum(proj_1 * proj_2, dim=-1)
    num_pairs = pos_cos.numel()
    num_selected = max(1, math.ceil(num_pairs * hard_ratio))
    selected_pos_cos = torch.topk(pos_cos, k=num_selected, largest=False).values

    # linear: mean(1 - cos(z_i, z_j))
    # square: mean((1 - cos(z_i, z_j))^2)
    penalties = 1.0 - selected_pos_cos
    if reg_form == "linear":
        reg_loss = penalties.mean()
    elif reg_form == "square":
        reg_loss = penalties.square().mean()
    else:
        raise ValueError(f"Unsupported reg_form: {reg_form}")

    return reg_loss, pos_cos, selected_pos_cos


@dataclass
class CompositeContrastiveLossConfig:
    temperature: float
    hard_ratio: float
    reg_form: str
    reg_weight: float


class CompositeContrastiveLoss(nn.Module):
    def __init__(self, config: CompositeContrastiveLossConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        proj_1: torch.Tensor,
        proj_2: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        infonce_loss = compute_infonce(proj_1, proj_2, temperature=self.config.temperature)
        pos_cos = torch.sum(proj_1 * proj_2, dim=-1)

        if self.config.reg_weight > 0:
            reg_loss, pos_cos, selected_pos_cos = compute_positive_regularization(
                proj_1,
                proj_2,
                hard_ratio=self.config.hard_ratio,
                reg_form=self.config.reg_form,
            )
            selected_pos_cos_mean = selected_pos_cos.mean()
        else:
            reg_loss = proj_1.new_zeros(())
            selected_pos_cos_mean = pos_cos.new_zeros(())

        total_loss = infonce_loss + self.config.reg_weight * reg_loss
        metrics = {
            "infonce_loss": infonce_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "selected_pos_cos_mean": selected_pos_cos_mean.detach(),
        }
        return total_loss, metrics, pos_cos.detach()
