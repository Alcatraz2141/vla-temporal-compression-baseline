from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = (pred - target).pow(2)
    if mask is None:
        return loss.mean()
    mask_f = mask.to(loss.dtype).unsqueeze(-1)
    return (loss * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def temporal_smoothness(actions: torch.Tensor) -> torch.Tensor:
    if actions.size(1) < 2:
        return actions.new_tensor(0.0)
    return (actions[:, 1:] - actions[:, :-1]).pow(2).mean()
