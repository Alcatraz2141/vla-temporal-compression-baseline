from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.libero_rollout import _load_action_stats
from evaluation.rollout_alignment_checks import demo_arrays, load_policy, predict_windows, task_file
from utils.config import load_config
from utils.seed import resolve_device, set_seed


DEMO_RE = re.compile(r"::(?P<rel_path>.+\.hdf5)::data/demo_(?P<demo_idx>\d+)$")


def split_demo_map(split_file: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for line in split_file.read_text(encoding="utf-8").splitlines():
        match = DEMO_RE.search(line.strip())
        if not match:
            continue
        task_name = Path(match.group("rel_path")).stem.removesuffix("_demo")
        out.setdefault(task_name, []).append(int(match.group("demo_idx")))
    return {task: sorted(set(indices)) for task, indices in sorted(out.items())}


def transition_masks(actions: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    expert_sign = actions[:, -1] >= 0.0
    transition = np.zeros(len(actions), dtype=np.bool_)
    transition[1:] = expert_sign[1:] != expert_sign[:-1]
    near_transition = np.zeros(len(actions), dtype=np.bool_)
    for idx in np.flatnonzero(transition):
        near_transition[max(0, idx - radius) : min(len(actions), idx + radius + 1)] = True
    return transition, near_transition


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def task_metrics(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    data_root: Path,
    task_name: str,
    demo_indices: list[int],
    action_stats: dict[str, Any] | None,
    device: torch.device,
    batch_size: int,
    near_radius: int,
) -> dict[str, Any]:
    h5_path = task_file(data_root, task_name)
    continuous_mse: list[float] = []
    continuous_mae: list[float] = []
    gripper_accuracy: list[float] = []
    pred_close_fraction: list[float] = []
    expert_close_fraction: list[float] = []
    transition_hits = 0
    transition_count = 0
    near_hits = 0
    near_count = 0
    frames = 0

    for demo_idx in demo_indices:
        images, states, actions = demo_arrays(h5_path, demo_idx)
        timesteps = list(range(len(actions)))
        preds = predict_windows(
            model,
            cfg,
            images,
            states,
            actions,
            timesteps,
            action_stats,
            task_name,
            device,
            batch_size,
        )
        pred_sign = preds[:, -1] >= 0.0
        expert_sign = actions[:, -1] >= 0.0
        transition, near_transition = transition_masks(actions, near_radius)
        cont_err = preds[:, :6] - actions[:, :6]

        continuous_mse.append(float(np.mean(cont_err**2)))
        continuous_mae.append(float(np.mean(np.abs(cont_err))))
        gripper_accuracy.append(float(np.mean(pred_sign == expert_sign)))
        pred_close_fraction.append(float(np.mean(~pred_sign)))
        expert_close_fraction.append(float(np.mean(~expert_sign)))
        transition_hits += int(np.sum(pred_sign[transition] == expert_sign[transition]))
        transition_count += int(np.sum(transition))
        near_hits += int(np.sum(pred_sign[near_transition] == expert_sign[near_transition]))
        near_count += int(np.sum(near_transition))
        frames += int(len(actions))

    return {
        "task_name": task_name,
        "demo_count": len(demo_indices),
        "demo_indices": " ".join(str(idx) for idx in demo_indices),
        "frames": frames,
        "continuous_mse": safe_mean(continuous_mse),
        "continuous_mae": safe_mean(continuous_mae),
        "gripper_sign_accuracy": safe_mean(gripper_accuracy),
        "transition_accuracy": float(transition_hits / transition_count) if transition_count else float("nan"),
        "transition_hits": transition_hits,
        "transition_count": transition_count,
        "near_transition_accuracy": float(near_hits / near_count) if near_count else float("nan"),
        "near_transition_hits": near_hits,
        "near_transition_count": near_count,
        "pred_close_fraction": safe_mean(pred_close_fraction),
        "expert_close_fraction": safe_mean(expert_close_fraction),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-task exact gripper transition diagnostics for corrected H=1 LIBERO policies.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", type=Path, default=Path("data/libero_long"))
    parser.add_argument("--results-path", type=Path, default=Path("results/per_task_transition_diagnostics.csv"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--near-radius", type=int, default=3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("model", {}).get("baseline") == "event_gated_memory":
        raise ValueError("This diagnostic currently supports reactive/sliding-window policies only.")
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(args.device or cfg.get("device", "auto"))
    model, ckpt_cfg = load_policy(args.config, args.checkpoint, device)
    action_stats = _load_action_stats(ckpt_cfg)
    split_file = args.split_file or Path(ckpt_cfg["data"]["episode_loader"].get("split_dir", "splits")) / f"{ckpt_cfg['data'].get('dataset', 'libero_long')}_{args.split}.txt"
    task_to_demos = split_demo_map(split_file)
    if not task_to_demos:
        raise ValueError(f"No LIBERO demo IDs found in split file: {split_file}")

    rows = [
        task_metrics(
            model,
            ckpt_cfg,
            args.data_root,
            task_name,
            demo_indices,
            action_stats,
            device,
            args.batch_size,
            args.near_radius,
        )
        for task_name, demo_indices in task_to_demos.items()
    ]
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_hits = sum(int(row["transition_hits"]) for row in rows)
    total_count = sum(int(row["transition_count"]) for row in rows)
    near_hits = sum(int(row["near_transition_hits"]) for row in rows)
    near_count = sum(int(row["near_transition_count"]) for row in rows)
    print(f"wrote {args.results_path}")
    print(f"overall transition accuracy: {total_hits}/{total_count} = {total_hits / total_count if total_count else float('nan'):.6f}")
    print(f"overall near-transition accuracy: {near_hits}/{near_count} = {near_hits / near_count if near_count else float('nan'):.6f}")


if __name__ == "__main__":
    main()
