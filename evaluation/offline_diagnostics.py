from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.data_loader import build_dataloader
from evaluation.eval import _model_forward_and_target, load_compatible_state_dict
from models.vla_baseline import build_model
from utils.config import load_config
from utils.seed import resolve_device, set_seed


def _zero_stats(action_dim: int, horizon: int) -> dict[str, Any]:
    return {
        "sq_all": 0.0,
        "abs_all": 0.0,
        "valid_steps": 0.0,
        "valid_elements": 0.0,
        "first_sq": 0.0,
        "first_abs": 0.0,
        "first_count": 0.0,
        "first_elements": 0.0,
        "horizon_sq": torch.zeros(horizon, dtype=torch.float64),
        "horizon_abs": torch.zeros(horizon, dtype=torch.float64),
        "horizon_count": torch.zeros(horizon, dtype=torch.float64),
        "dim_sq": torch.zeros(action_dim, dtype=torch.float64),
        "dim_abs": torch.zeros(action_dim, dtype=torch.float64),
        "dim_count": torch.zeros(action_dim, dtype=torch.float64),
        "gripper_correct": 0.0,
        "gripper_count": 0.0,
        "gripper_transition_correct": 0.0,
        "gripper_transition_count": 0.0,
    }


def _update_stats(stats: dict[str, Any], pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    mask = mask.detach().cpu().bool()

    err = pred - target
    sq = err.pow(2).double()
    abs_err = err.abs().double()
    mask_f = mask.double()
    elem_mask = mask_f.unsqueeze(-1)
    action_dim = pred.size(-1)

    stats["sq_all"] += float((sq * elem_mask).sum())
    stats["abs_all"] += float((abs_err * elem_mask).sum())
    stats["valid_steps"] += float(mask_f.sum())
    stats["valid_elements"] += float(mask_f.sum() * action_dim)

    first_mask = mask[:, 0]
    if first_mask.any():
        first_elem = first_mask.double().unsqueeze(-1)
        stats["first_sq"] += float((sq[:, 0] * first_elem).sum())
        stats["first_abs"] += float((abs_err[:, 0] * first_elem).sum())
        stats["first_count"] += float(first_mask.double().sum())
        stats["first_elements"] += float(first_mask.double().sum() * action_dim)

    stats["horizon_sq"] += (sq * elem_mask).sum(dim=(0, 2))
    stats["horizon_abs"] += (abs_err * elem_mask).sum(dim=(0, 2))
    stats["horizon_count"] += mask_f.sum(dim=0) * action_dim

    stats["dim_sq"] += (sq * elem_mask).sum(dim=(0, 1))
    stats["dim_abs"] += (abs_err * elem_mask).sum(dim=(0, 1))
    stats["dim_count"] += mask_f.sum()

    if action_dim >= 7:
        pred_sign = pred[..., -1] >= 0.0
        target_sign = target[..., -1] >= 0.0
        stats["gripper_correct"] += float(((pred_sign == target_sign) & mask).sum())
        stats["gripper_count"] += float(mask.sum())

        if pred.size(1) > 1:
            pair_mask = mask[:, 1:] & mask[:, :-1]
            pred_transition = pred_sign[:, 1:] != pred_sign[:, :-1]
            target_transition = target_sign[:, 1:] != target_sign[:, :-1]
            stats["gripper_transition_correct"] += float(((pred_transition == target_transition) & pair_mask).sum())
            stats["gripper_transition_count"] += float(pair_mask.sum())


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _list_div(num: torch.Tensor, den: torch.Tensor) -> list[float]:
    out = torch.full_like(num, float("nan"), dtype=torch.float64)
    valid = den > 0
    out[valid] = num[valid] / den[valid]
    return [float(value) for value in out.tolist()]


def _finalize(stats: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "mse_per_element": _safe_div(stats["sq_all"], stats["valid_elements"]),
        "mae_per_element": _safe_div(stats["abs_all"], stats["valid_elements"]),
        "legacy_mse_per_timestep": _safe_div(stats["sq_all"], stats["valid_steps"]),
        "legacy_mae_per_timestep": _safe_div(stats["abs_all"], stats["valid_steps"]),
        "first_action_mse_per_element": _safe_div(stats["first_sq"], stats["first_elements"]),
        "first_action_mae_per_element": _safe_div(stats["first_abs"], stats["first_elements"]),
        "valid_action_steps": int(stats["valid_steps"]),
        "valid_first_actions": int(stats["first_count"]),
        "per_horizon_mse": _list_div(stats["horizon_sq"], stats["horizon_count"]),
        "per_horizon_mae": _list_div(stats["horizon_abs"], stats["horizon_count"]),
        "per_dim_mse": _list_div(stats["dim_sq"], stats["dim_count"]),
        "per_dim_mae": _list_div(stats["dim_abs"], stats["dim_count"]),
        "gripper_sign_accuracy": _safe_div(stats["gripper_correct"], stats["gripper_count"]),
        "gripper_transition_accuracy": _safe_div(
            stats["gripper_transition_correct"],
            stats["gripper_transition_count"],
        ),
        "gripper_transition_count": int(stats["gripper_transition_count"]),
    }

    per_dim_mse = metrics["per_dim_mse"]
    per_dim_mae = metrics["per_dim_mae"]
    if len(per_dim_mse) >= 6:
        metrics["position_mse"] = sum(per_dim_mse[:3]) / 3.0
        metrics["position_mae"] = sum(per_dim_mae[:3]) / 3.0
        metrics["rotation_mse"] = sum(per_dim_mse[3:6]) / 3.0
        metrics["rotation_mae"] = sum(per_dim_mae[3:6]) / 3.0
    if len(per_dim_mse) >= 7:
        metrics["gripper_mse"] = per_dim_mse[6]
        metrics["gripper_mae"] = per_dim_mae[6]
    return metrics


@torch.no_grad()
def run_diagnostics(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    stats: dict[str, Any] | None = None
    for batch in tqdm(loader, desc="diagnostics", leave=False):
        pred, target, mask = _model_forward_and_target(model, batch, device)
        if stats is None:
            stats = _zero_stats(action_dim=pred.size(-1), horizon=pred.size(1))
        _update_stats(stats, pred, target, mask)
    if stats is None:
        raise ValueError("No batches available for diagnostics.")
    return _finalize(stats)


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _jsonify_lists(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        out[key] = json.dumps(value) if isinstance(value, list) else value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rollout-aligned offline diagnostics for LIBERO action prediction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default=None, help="Dataset split to evaluate. Defaults to config data.val_split.")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cpu or mps.")
    parser.add_argument("--results-path", type=Path, default=Path("results/offline_diagnostics.csv"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(args.device or cfg.get("device", "auto"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = checkpoint.get("config", cfg)
    ckpt_cfg["model"]["baseline"] = cfg["model"].get("baseline", ckpt_cfg["model"].get("baseline", "sliding_window"))
    ckpt_cfg["model"]["run_name"] = cfg["model"].get(
        "run_name",
        ckpt_cfg["model"].get("run_name", ckpt_cfg["model"]["baseline"]),
    )

    split = args.split or ckpt_cfg["data"].get("val_split", "val")
    loader = build_dataloader(ckpt_cfg, split, shuffle=False)
    model = build_model(ckpt_cfg, int(checkpoint["state_dim"]), int(checkpoint["action_dim"])).to(device)
    load_compatible_state_dict(model, checkpoint["model"])

    metrics = run_diagnostics(model, loader, device)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline": ckpt_cfg["model"].get("run_name", ckpt_cfg["model"].get("baseline", "policy")),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "split": split,
        "seed": ckpt_cfg.get("seed", 42),
        **metrics,
    }
    append_row(args.results_path, _jsonify_lists(row))
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
