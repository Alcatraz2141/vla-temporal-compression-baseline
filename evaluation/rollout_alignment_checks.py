from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.eval import load_compatible_state_dict
from evaluation.libero_rollout import _load_action_stats, unnormalize_action_chunk
from models.vla_baseline import build_model
from utils.config import load_config
from utils.language import language_ids
from utils.seed import resolve_device, set_seed

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def task_file(data_root: Path, task_name: str) -> Path:
    matches = sorted(data_root.glob(f"**/{task_name}_demo.hdf5"))
    if not matches:
        raise FileNotFoundError(f"No HDF5 file matched task {task_name!r} under {data_root}")
    return matches[0]


def load_policy(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_cfg = checkpoint.get("config", cfg)
    ckpt_cfg["model"]["baseline"] = cfg["model"].get("baseline", ckpt_cfg["model"].get("baseline", "sliding_window"))
    ckpt_cfg["model"]["run_name"] = cfg["model"].get(
        "run_name",
        ckpt_cfg["model"].get("run_name", ckpt_cfg["model"]["baseline"]),
    )
    model = build_model(ckpt_cfg, int(checkpoint["state_dim"]), int(checkpoint["action_dim"])).to(device)
    load_compatible_state_dict(model, checkpoint["model"])
    model.eval()
    return model, ckpt_cfg


def normalize_image_batch(images: np.ndarray, image_normalization: str | None) -> torch.Tensor:
    arr = images.astype(np.float32) / 255.0
    if image_normalization in {"imagenet", "resnet", "imageNet"}:
        arr = (arr - IMAGENET_MEAN.reshape(1, 1, 1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 1, 1, 3)
    elif image_normalization not in {None, "", "none"}:
        raise ValueError(f"Unsupported image_normalization={image_normalization!r}")
    return torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous()


def action_norm_arrays(action_stats: dict[str, Any] | None, action_dim: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if action_stats is None:
        return np.zeros(action_dim, dtype=np.float32), np.ones(action_dim, dtype=np.float32), []
    mean = np.asarray(action_stats.get("mean", [0.0] * action_dim), dtype=np.float32).reshape(-1)
    std = np.asarray(action_stats.get("std", [1.0] * action_dim), dtype=np.float32).reshape(-1)
    if mean.size != action_dim:
        mean = np.pad(mean[:action_dim], (0, max(action_dim - mean.size, 0)), constant_values=0.0)[:action_dim]
    if std.size != action_dim:
        std = np.pad(std[:action_dim], (0, max(action_dim - std.size, 0)), constant_values=1.0)[:action_dim]
    dims = [int(dim) for dim in action_stats.get("normalize_dims", list(range(action_dim))) if 0 <= int(dim) < action_dim]
    return mean, np.clip(std, 1e-6, None), dims


def normalize_actions(actions: np.ndarray, action_stats: dict[str, Any] | None) -> np.ndarray:
    out = actions.astype(np.float32, copy=True)
    if action_stats is None:
        return out
    mean, std, dims = action_norm_arrays(action_stats, out.shape[-1])
    if dims:
        out[..., dims] = (out[..., dims] - mean[dims]) / std[dims]
    return out


def demo_arrays(h5_path: Path, demo_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        demo = h5[f"data/demo_{demo_idx}"]
        obs = demo["obs"]
        images = np.asarray(obs["agentview_rgb"])
        states = np.concatenate(
            [np.asarray(obs["ee_pos"]), np.asarray(obs["ee_ori"]), np.asarray(obs["gripper_states"])],
            axis=-1,
        ).astype(np.float32)
        actions = np.asarray(demo["actions"]).astype(np.float32)
    length = min(len(images), len(states), len(actions))
    return images[:length], states[:length], actions[:length]


def make_recent(actions: np.ndarray, t: int, k_recent: int) -> tuple[np.ndarray, np.ndarray]:
    recent_idx = np.arange(max(0, t - k_recent + 1), t + 1, dtype=np.int64)
    mask = np.ones(len(recent_idx), dtype=np.bool_)
    if len(recent_idx) < k_recent:
        pad = k_recent - len(recent_idx)
        recent_idx = np.concatenate([np.zeros(pad, dtype=np.int64), recent_idx])
        mask = np.concatenate([np.zeros(pad, dtype=np.bool_), mask])
    prev_actions = np.zeros_like(actions, dtype=np.float32)
    if len(actions) > 1:
        prev_actions[1:] = actions[:-1]
    recent_actions = prev_actions[recent_idx].astype(np.float32)
    recent_actions[~mask] = 0.0
    return recent_idx, recent_actions


@torch.no_grad()
def predict_windows(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    images: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    timesteps: list[int],
    action_stats: dict[str, Any] | None,
    language: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    k_recent = int(cfg["data"].get("K_recent", cfg["data"].get("T_obs", 8)))
    image_normalization = cfg.get("data", {}).get("image_normalization", cfg.get("data", {}).get("normalization", {}).get("images"))
    preds: list[np.ndarray] = []
    for start in range(0, len(timesteps), batch_size):
        batch_ts = timesteps[start : start + batch_size]
        idxs = []
        recent_actions = []
        for t in batch_ts:
            idx, acts = make_recent(actions, t, k_recent)
            idxs.append(idx)
            recent_actions.append(acts)
        idx_arr = np.stack(idxs, axis=0)
        recent_obs = normalize_image_batch(images[idx_arr], image_normalization).to(device)
        recent_states = torch.from_numpy(states[idx_arr].astype(np.float32)).to(device)
        recent_actions_np = normalize_actions(np.stack(recent_actions, axis=0), action_stats)
        recent_actions_t = torch.from_numpy(recent_actions_np).to(device)
        kwargs: dict[str, Any] = {}
        if bool(getattr(model, "use_language", False)):
            vocab_size = int(getattr(model, "language_embedding").num_embeddings)
            kwargs["language_ids"] = language_ids([language] * len(batch_ts), vocab_size, device)
        pred = model(images=recent_obs, states=recent_states, actions=recent_actions_t, **kwargs)
        pred_np = pred[:, 0].detach().cpu().numpy().astype(np.float32)
        pred_np = unnormalize_action_chunk(pred_np[:, None, :], action_stats)[:, 0, :]
        preds.append(pred_np)
    return np.concatenate(preds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check HDF5 expert/action alignment for a rollout-facing checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/libero_long"))
    parser.add_argument("--task-name", default="STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy")
    parser.add_argument("--demo-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(args.device or cfg.get("device", "auto"))
    model, ckpt_cfg = load_policy(args.config, args.checkpoint, device)
    action_stats = _load_action_stats(ckpt_cfg)
    h5_path = task_file(args.data_root, args.task_name)
    images, states, actions = demo_arrays(h5_path, args.demo_idx)
    language = h5_path.stem.removesuffix("_demo")

    first_pred = predict_windows(model, ckpt_cfg, images, states, actions, [0], action_stats, language, device, args.batch_size)[0]
    first_expert = actions[0]
    first_raw_err = first_pred - first_expert

    timesteps = list(range(len(actions)))
    preds = predict_windows(model, ckpt_cfg, images, states, actions, timesteps, action_stats, language, device, args.batch_size)
    pred_sign = preds[:, -1] >= 0.0
    expert_sign = actions[:, -1] >= 0.0
    transition = np.zeros(len(actions), dtype=np.bool_)
    transition[1:] = expert_sign[1:] != expert_sign[:-1]
    near_transition = np.zeros(len(actions), dtype=np.bool_)
    for idx in np.flatnonzero(transition):
        near_transition[max(0, idx - 3) : min(len(actions), idx + 4)] = True

    cont_err = preds[:, :6] - actions[:, :6]
    result = {
        "task_name": args.task_name,
        "demo_idx": args.demo_idx,
        "hdf5_path": str(h5_path),
        "length": int(len(actions)),
        "first_pred_raw": first_pred.tolist(),
        "first_expert_raw": first_expert.tolist(),
        "first_error_raw": first_raw_err.tolist(),
        "first_continuous_l2": float(np.linalg.norm(first_raw_err[:6])),
        "first_gripper_match": bool((first_pred[-1] >= 0.0) == (first_expert[-1] >= 0.0)),
        "continuous_mse_raw": float(np.mean(cont_err**2)),
        "continuous_mae_raw": float(np.mean(np.abs(cont_err))),
        "gripper_sign_accuracy_full_demo": float(np.mean(pred_sign == expert_sign)),
        "expert_gripper_transition_count": int(np.sum(transition)),
        "gripper_sign_accuracy_at_transitions": float(np.mean(pred_sign[transition] == expert_sign[transition])) if np.any(transition) else float("nan"),
        "gripper_sign_accuracy_near_transitions": float(np.mean(pred_sign[near_transition] == expert_sign[near_transition])) if np.any(near_transition) else float("nan"),
        "predicted_close_fraction": float(np.mean(~pred_sign)),
        "expert_close_fraction": float(np.mean(~expert_sign)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
