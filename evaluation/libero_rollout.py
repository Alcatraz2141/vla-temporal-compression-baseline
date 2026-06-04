from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import deque
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.vla_baseline import EventGatedMemoryVLA, build_model
from utils.config import load_config
from utils.language import language_ids
from utils.seed import resolve_device, set_seed

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def load_compatible_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    patched = dict(state_dict)
    gate_key = "gate.0.weight"
    if gate_key in patched and gate_key in model_state and patched[gate_key].shape != model_state[gate_key].shape:
        old_weight = patched[gate_key]
        new_weight = model_state[gate_key].clone()
        cols = min(old_weight.shape[1], new_weight.shape[1])
        new_weight[:, :cols] = old_weight[:, :cols]
        patched[gate_key] = new_weight
    result = model.load_state_dict(patched, strict=False)
    allowed_prefixes = (
        "phase_embedding.",
        "secured_embedding.",
        "placement_ready_embedding.",
        "memory_pos_embedding",
        "memory_pool_query",
        "memory_gate.",
    )
    unexpected = [key for key in result.unexpected_keys if not key.startswith(allowed_prefixes)]
    missing = [key for key in result.missing_keys if not key.startswith(allowed_prefixes)]
    if unexpected or missing:
        raise RuntimeError(f"Incompatible checkpoint keys. missing={missing}, unexpected={unexpected}")


def _patch_torch_load_for_libero_init_states() -> None:
    """LIBERO init-state files are trusted local files and need full unpickling on PyTorch 2.6+."""
    original_load = torch.load

    def patched_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load  # type: ignore[assignment]


def _import_libero() -> tuple[Any, Any, Any]:
    _patch_torch_load_for_libero_init_states()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return benchmark, get_libero_path, OffScreenRenderEnv


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    from robosuite.utils.transform_utils import quat2axisangle

    return np.asarray(quat2axisangle(quat), dtype=np.float32)


def obs_to_image(obs: dict[str, Any], camera_key: str) -> np.ndarray:
    image = np.asarray(obs[camera_key])
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def obs_to_state(obs: dict[str, Any]) -> np.ndarray:
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    ori = _quat_to_axis_angle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1))
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    return np.concatenate([pos, ori, gripper], axis=0).astype(np.float32)


def image_tensor(images: list[np.ndarray], device: torch.device, image_normalization: str | None = None) -> torch.Tensor:
    arr = np.stack(images, axis=0).astype(np.float32) / 255.0
    if image_normalization in {"imagenet", "resnet", "imageNet"}:
        arr = (arr - IMAGENET_MEAN.reshape(1, 1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 1, 3)
    elif image_normalization not in {None, "", "none"}:
        raise ValueError(f"Unsupported image_normalization={image_normalization!r}. Expected 'imagenet' or 'none'.")
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)
    return tensor.to(device)


def vector_tensor(vectors: list[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack(vectors, axis=0).astype(np.float32)).unsqueeze(0).to(device)


def mask_tensor(mask: list[bool], device: torch.device) -> torch.Tensor:
    return torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)


def _phase_boundaries_from_config(cfg: dict[str, Any]) -> list[int]:
    episode_cfg = cfg.get("data", {}).get("episode_loader", {})
    boundaries = episode_cfg.get("phase_boundaries")
    if boundaries is None:
        placement_start = episode_cfg.get("placement_start_timestep")
        boundaries = [] if placement_start is None else [placement_start]
    return sorted(int(boundary) for boundary in boundaries)


def _phase_ids(indices: np.ndarray, boundaries: list[int], device: torch.device) -> torch.Tensor:
    phases = np.zeros(indices.shape, dtype=np.int64)
    for boundary in boundaries:
        phases += indices >= boundary
    return torch.from_numpy(phases).unsqueeze(0).to(device)


def _derived_signal_arrays(
    actions: list[np.ndarray],
    states: list[np.ndarray],
    indices: np.ndarray,
    boundaries: list[int],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    episode_cfg = cfg.get("data", {}).get("episode_loader", {})
    secured_delay = max(int(episode_cfg.get("secured_delay_steps", 8)), 0)
    secured_min_positive = max(int(episode_cfg.get("secured_min_positive_steps", 4)), 1)
    secured_window = max(int(episode_cfg.get("secured_history_window", 8)), 1)
    ready_phase = max(int(episode_cfg.get("placement_ready_phase", 1)), 0)
    ready_delay = max(int(episode_cfg.get("placement_ready_delay_steps", 20)), 0)
    pose_threshold = float(episode_cfg.get("placement_ready_pose_stability_threshold", 0.06))
    action_threshold = float(episode_cfg.get("placement_ready_action_stability_threshold", 0.75))
    stability_window = max(int(episode_cfg.get("placement_ready_stability_window", 4)), 1)

    action_arr = np.asarray(actions, dtype=np.float32)
    state_arr = np.asarray(states, dtype=np.float32)
    positive = action_arr[:, -1] >= 0.0
    positive_steps = np.flatnonzero(positive)
    first_positive = int(positive_steps[0]) if positive_steps.size else None
    secured = np.zeros(indices.shape, dtype=np.int64)
    ready = np.zeros(indices.shape, dtype=np.int64)
    phases = np.zeros(indices.shape, dtype=np.int64)
    for boundary in boundaries:
        phases += indices >= boundary
    for out_i, raw_index in enumerate(indices):
        if first_positive is None:
            continue
        index = min(max(int(raw_index), 0), len(action_arr) - 1)
        history_start = max(0, index - secured_window + 1)
        positive_count = int(positive[history_start : index + 1].sum())
        is_secured = index >= first_positive + secured_delay and positive_count >= secured_min_positive
        secured[out_i] = int(is_secured)
        if not is_secured:
            continue
        stable_start = max(0, index - stability_window + 1)
        state_deltas = np.linalg.norm(np.diff(state_arr[stable_start : index + 1, :3], axis=0), axis=-1)
        action_norms = np.linalg.norm(action_arr[stable_start : index + 1, :6], axis=-1)
        pose_stable = state_deltas.size == 0 or float(state_deltas.max()) <= pose_threshold
        action_stable = action_norms.size == 0 or float(action_norms.mean()) <= action_threshold
        ready[out_i] = int(phases[out_i] >= ready_phase and index >= first_positive + ready_delay and pose_stable and action_stable)
    return secured, ready


def _load_action_stats(cfg: dict[str, Any]) -> dict[str, Any] | None:
    normalization = cfg.get("data", {}).get("normalization", {})
    if not bool(normalization.get("actions", False)):
        return None
    stats_path = normalization.get("stats_path")
    if not stats_path:
        raise ValueError("data.normalization.actions=true requires data.normalization.stats_path for rollout.")
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Action normalization stats not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats.get("actions", stats)


def _action_norm_arrays(action_stats: dict[str, Any], action_dim: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    mean = np.asarray(action_stats.get("mean", [0.0] * action_dim), dtype=np.float32).reshape(-1)
    std = np.asarray(action_stats.get("std", [1.0] * action_dim), dtype=np.float32).reshape(-1)
    if mean.size != action_dim:
        mean = np.pad(mean[:action_dim], (0, max(action_dim - mean.size, 0)), constant_values=0.0)[:action_dim]
    if std.size != action_dim:
        std = np.pad(std[:action_dim], (0, max(action_dim - std.size, 0)), constant_values=1.0)[:action_dim]
    std = np.clip(std, 1e-6, None)
    dims = action_stats.get("normalize_dims", list(range(action_dim)))
    valid_dims = [int(dim) for dim in dims if 0 <= int(dim) < action_dim]
    return mean, std, valid_dims


def normalize_actions(actions: list[np.ndarray], action_stats: dict[str, Any] | None, action_dim: int) -> list[np.ndarray]:
    if action_stats is None:
        return actions
    mean, std, dims = _action_norm_arrays(action_stats, action_dim)
    out = []
    for action in actions:
        normalized = np.asarray(action, dtype=np.float32).copy()
        if dims:
            normalized[dims] = (normalized[dims] - mean[dims]) / std[dims]
        out.append(normalized)
    return out


def unnormalize_action_chunk(actions: np.ndarray, action_stats: dict[str, Any] | None) -> np.ndarray:
    if action_stats is None:
        return actions
    action_dim = actions.shape[-1]
    mean, std, dims = _action_norm_arrays(action_stats, action_dim)
    out = actions.astype(np.float32, copy=True)
    if dims:
        out[..., dims] = out[..., dims] * std[dims] + mean[dims]
    return out


class OnlineHistory:
    def __init__(self, k_recent: int, older_len: int, action_dim: int, first_image: np.ndarray, first_state: np.ndarray) -> None:
        self.k_recent = k_recent
        self.older_len = older_len
        self.action_dim = action_dim
        self.first_image = first_image
        self.first_state = first_state
        self.images: list[np.ndarray] = [first_image]
        self.states: list[np.ndarray] = [first_state]
        self.prev_actions: list[np.ndarray] = [np.zeros(action_dim, dtype=np.float32)]

    def append(self, image: np.ndarray, state: np.ndarray, action: np.ndarray) -> None:
        self.images.append(image)
        self.states.append(state)
        self.prev_actions.append(action.astype(np.float32))

    def recent(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[bool]]:
        start = max(0, len(self.images) - self.k_recent)
        images = self.images[start:]
        states = self.states[start:]
        actions = self.prev_actions[start:]
        valid = [True] * len(images)
        pad = self.k_recent - len(images)
        if pad > 0:
            images = [self.first_image] * pad + images
            states = [self.first_state] * pad + states
            actions = [np.zeros(self.action_dim, dtype=np.float32)] * pad + actions
            valid = [False] * pad + valid
        return images, states, actions, valid

    def older(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[bool]]:
        cutoff = max(0, len(self.images) - self.k_recent)
        start = max(0, cutoff - self.older_len)
        images = self.images[start:cutoff]
        states = self.states[start:cutoff]
        actions = self.prev_actions[start:cutoff]
        valid = [True] * len(images)
        pad = self.older_len - len(images)
        zero_image = np.zeros_like(self.first_image)
        zero_state = np.zeros_like(self.first_state)
        zero_action = np.zeros(self.action_dim, dtype=np.float32)
        if pad > 0:
            images = [zero_image] * pad + images
            states = [zero_state] * pad + states
            actions = [zero_action] * pad + actions
            valid = [False] * pad + valid
        return images, states, actions, valid


@torch.no_grad()
def predict_chunk(
    model: torch.nn.Module,
    history: OnlineHistory,
    device: torch.device,
    clip_action: bool,
    discrete_gripper: bool,
    action_stats: dict[str, Any] | None = None,
    image_normalization: str | None = None,
    language: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    recent_images, recent_states, recent_actions, recent_mask = history.recent()
    recent_actions = normalize_actions(recent_actions, action_stats, history.action_dim)
    kwargs: dict[str, Any] = {}
    if bool(getattr(model, "use_language", False)):
        vocab_size = int(getattr(model, "language_embedding").num_embeddings)
        kwargs["language_ids"] = language_ids([language or ""], vocab_size, device)
    if bool(getattr(model, "use_phase", False)):
        boundaries = _phase_boundaries_from_config(cfg or {})
        current_step = len(history.images) - 1
        raw_recent = np.arange(max(0, current_step - history.k_recent + 1), current_step + 1, dtype=np.int64)
        pad = history.k_recent - len(raw_recent)
        if pad > 0:
            recent_indices = np.concatenate([np.zeros(pad, dtype=np.int64), raw_recent])
        else:
            recent_indices = raw_recent[-history.k_recent :]
        target_indices = current_step + np.arange(int(getattr(model, "T_action", 1)), dtype=np.int64)
        recent_phase_ids = _phase_ids(recent_indices, boundaries, device)
        target_phase_ids = _phase_ids(target_indices, boundaries, device)
        kwargs["recent_phase_ids"] = recent_phase_ids
        kwargs["target_phase_ids"] = target_phase_ids
        if bool(getattr(model, "use_object_signals", False)):
            recent_secured, recent_ready = _derived_signal_arrays(history.prev_actions, history.states, recent_indices, boundaries, cfg or {})
            target_secured, target_ready = _derived_signal_arrays(history.prev_actions, history.states, target_indices, boundaries, cfg or {})
            kwargs["recent_secured_ids"] = torch.from_numpy(recent_secured).unsqueeze(0).to(device)
            kwargs["target_secured_ids"] = torch.from_numpy(target_secured).unsqueeze(0).to(device)
            kwargs["recent_placement_ready_ids"] = torch.from_numpy(recent_ready).unsqueeze(0).to(device)
            kwargs["target_placement_ready_ids"] = torch.from_numpy(target_ready).unsqueeze(0).to(device)
        else:
            recent_secured = np.zeros_like(recent_indices, dtype=np.int64)
            target_secured = np.zeros_like(target_indices, dtype=np.int64)
            recent_ready = np.zeros_like(recent_indices, dtype=np.int64)
            target_ready = np.zeros_like(target_indices, dtype=np.int64)
        signal_info = {
            "current_phase_id": int(recent_phase_ids[0, -1].detach().cpu()),
            "target_phase_ids": target_phase_ids.squeeze(0).detach().cpu().numpy().astype(np.int64),
            "current_secured": int(recent_secured[-1]),
            "target_secured": target_secured.astype(np.int64),
            "current_placement_ready": int(recent_ready[-1]),
            "target_placement_ready": target_ready.astype(np.int64),
        }
    else:
        signal_info = {
            "current_phase_id": 0,
            "target_phase_ids": np.zeros(int(getattr(model, "T_action", 1)), dtype=np.int64),
            "current_secured": 0,
            "target_secured": np.zeros(int(getattr(model, "T_action", 1)), dtype=np.int64),
            "current_placement_ready": 0,
            "target_placement_ready": np.zeros(int(getattr(model, "T_action", 1)), dtype=np.int64),
        }
    if isinstance(model, EventGatedMemoryVLA) or bool(getattr(model, "use_event_memory", False)):
        older_images, older_states, older_actions, older_mask = history.older()
        older_actions = normalize_actions(older_actions, action_stats, history.action_dim)
        if isinstance(model, EventGatedMemoryVLA):
            pred = model(
                recent_obs=image_tensor(recent_images, device, image_normalization),
                recent_actions=vector_tensor(recent_actions, device),
                recent_states=vector_tensor(recent_states, device),
                older_obs=image_tensor(older_images, device, image_normalization),
                older_actions=vector_tensor(older_actions, device),
                older_states=vector_tensor(older_states, device),
                recent_mask=mask_tensor(recent_mask, device),
                older_mask=mask_tensor(older_mask, device),
                **kwargs,
            )
        else:
            pred = model(
                images=image_tensor(recent_images, device, image_normalization),
                states=vector_tensor(recent_states, device),
                actions=vector_tensor(recent_actions, device),
                older_obs=image_tensor(older_images, device, image_normalization),
                older_actions=vector_tensor(older_actions, device),
                older_states=vector_tensor(older_states, device),
                recent_mask=mask_tensor(recent_mask, device),
                older_mask=mask_tensor(older_mask, device),
                **kwargs,
            )
    else:
        images = image_tensor(recent_images, device, image_normalization)
        states = vector_tensor(recent_states, device)
        actions = vector_tensor(recent_actions, device)
        if hasattr(model, "sample_actions"):
            pred = model.sample_actions(
                images=images,
                states=states,
                actions=actions,
                **kwargs,
            )
        else:
            pred = model(
                images=images,
                states=states,
                actions=actions,
                **kwargs,
            )
    actions = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
    actions = unnormalize_action_chunk(actions, action_stats)
    raw_actions = actions.copy()
    if clip_action:
        actions = np.clip(actions, -1.0, 1.0)
    if discrete_gripper and actions.shape[-1] >= 7:
        actions[:, -1] = np.where(actions[:, -1] >= 0.0, 1.0, -1.0)
    signal_info["raw_actions"] = raw_actions
    return actions, signal_info


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def load_policy(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], int, int]:
    cfg = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_cfg = checkpoint.get("config", cfg)
    ckpt_cfg["model"]["baseline"] = cfg["model"].get("baseline", ckpt_cfg["model"].get("baseline", "sliding_window"))
    ckpt_cfg["model"]["run_name"] = cfg["model"].get("run_name", ckpt_cfg["model"].get("run_name", ckpt_cfg["model"]["baseline"]))
    model = build_model(ckpt_cfg, int(checkpoint["state_dim"]), int(checkpoint["action_dim"])).to(device)
    load_compatible_state_dict(model, checkpoint["model"])
    model.eval()
    return model, ckpt_cfg, int(checkpoint["state_dim"]), int(checkpoint["action_dim"])


def parse_tasks(value: str | None, total: int) -> list[int]:
    if value is None or value == "all":
        return list(range(total))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _normalize_task_name(name: str) -> str:
    return name.removesuffix("_demo").lower()


def _split_demo_index(demo_key: str) -> int | None:
    name = demo_key.rsplit("/", maxsplit=1)[-1]
    if not name.startswith("demo_"):
        return None
    try:
        return int(name.removeprefix("demo_"))
    except ValueError:
        return None


def split_init_indices(split_file: Path, task_name: str) -> list[int]:
    """Return init-state indices for a LIBERO task that appear in an episode split file."""
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    wanted_task = _normalize_task_name(task_name)
    indices: list[int] = []
    seen: set[int] = set()
    for line in split_file.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry:
            continue
        parts = entry.split("::")
        if len(parts) < 3:
            continue
        rel_path = parts[-2]
        demo_key = parts[-1]
        file_task = _normalize_task_name(Path(rel_path).stem)
        if file_task != wanted_task:
            continue
        index = _split_demo_index(demo_key)
        if index is None or index in seen:
            continue
        seen.add(index)
        indices.append(index)
    return sorted(indices)


def selected_episode_indices(
    split_file: Path | None,
    task_name: str,
    episodes_per_task: int,
    init_state_count: int,
) -> list[int]:
    if split_file is None:
        return list(range(min(episodes_per_task, init_state_count)))
    indices = [index for index in split_init_indices(split_file, task_name) if 0 <= index < init_state_count]
    return indices[:episodes_per_task]


def find_libero_hdf5(data_root: Path, task_name: str) -> Path:
    candidates = [
        data_root / "libero_10" / f"{task_name}_demo.hdf5",
        data_root / f"{task_name}_demo.hdf5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = [Path(path) for path in glob.glob(str(data_root / "**" / f"{task_name}_demo.hdf5"), recursive=True)]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find HDF5 demo file for task {task_name!r} under {data_root}.")


def load_expert_actions(data_root: Path, task_name: str, episode_idx: int, action_dim: int) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Expert-prefix diagnostics require h5py in the rollout environment.") from exc

    hdf5_path = find_libero_hdf5(data_root, task_name)
    demo_key = f"data/demo_{episode_idx}"
    with h5py.File(hdf5_path, "r") as h5:
        if demo_key not in h5:
            raise KeyError(f"{demo_key} not found in {hdf5_path}.")
        actions = np.asarray(h5[demo_key]["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != action_dim:
        raise ValueError(f"Expected expert actions [T,{action_dim}] in {hdf5_path}:{demo_key}, got {actions.shape}.")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained PyTorch policies in the official LIBERO simulator.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--tasks", default="0", help="Comma-separated task ids, or 'all'.")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Optional episode split file. When set, rollout init states are selected by demo_N indices from this split.",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--camera-key", default="agentview_image")
    parser.add_argument("--results-path", type=Path, default=Path("results/libero_rollouts.csv"))
    parser.add_argument("--clip-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--discrete-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--execute-horizon", type=int, default=1, help="Number of predicted chunk actions to execute before replanning.")
    parser.add_argument("--temporal-ensemble", action="store_true", help="Average overlapping predicted action chunks, favoring recent predictions.")
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.3)
    parser.add_argument("--video-dir", type=Path, default=None, help="Optional directory for rollout MP4 videos.")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-every", type=int, default=1, help="Record every N simulator steps when --video-dir is set.")
    parser.add_argument("--trace-path", type=Path, default=None, help="Optional CSV path for per-step rollout state/action traces.")
    parser.add_argument(
        "--expert-prefix-steps",
        type=int,
        default=0,
        help="Execute this many matching demonstration actions before handing control to the policy.",
    )
    parser.add_argument(
        "--expert-data-root",
        type=Path,
        default=None,
        help="LIBERO HDF5 root for --expert-prefix-steps. Defaults to data.episode_loader.root from the config.",
    )
    parser.add_argument("--dry-run-selection", action="store_true", help="Print split-selected demo indices and exit before importing LIBERO.")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    set_seed(args.seed)
    if args.dry_run_selection:
        if args.split_file is None:
            print({"split_file": None, "tasks": parse_tasks(args.tasks, 10), "note": "No split file provided; rollout will use first N init states."})
            return
        task_names_by_suite = {
            "libero_10": [
                "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
                "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
                "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
                "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
                "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
                "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
                "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
                "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
                "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
                "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
            ],
        }
        task_names = task_names_by_suite.get(args.suite)
        if task_names is None:
            raise ValueError(f"Dry-run task names are not hard-coded for suite={args.suite!r}.")
        task_ids = parse_tasks(args.tasks, len(task_names))
        for task_id in task_ids:
            task_name = task_names[task_id]
            indices = split_init_indices(args.split_file, task_name)[: args.episodes_per_task]
            print({"suite": args.suite, "task_id": task_id, "task_name": task_name, "split_file": str(args.split_file), "indices": indices})
        return

    device = resolve_device(args.device)
    benchmark, get_libero_path, OffScreenRenderEnv = _import_libero()
    model, cfg, state_dim, action_dim = load_policy(args.config, args.checkpoint, device)
    action_stats = _load_action_stats(cfg)
    image_normalization = cfg.get("data", {}).get("image_normalization", cfg.get("data", {}).get("normalization", {}).get("images"))
    if state_dim != 8 or action_dim != 7:
        raise ValueError(f"Expected LIBERO state_dim=8 and action_dim=7, got state_dim={state_dim}, action_dim={action_dim}.")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = parse_tasks(args.tasks, suite.n_tasks)
    k_recent = int(cfg["data"].get("K_recent", cfg["data"].get("T_obs", 8)))
    older_len = int(cfg.get("memory", {}).get("chunk_size", 4)) * int(cfg.get("memory", {}).get("max_memory_tokens", 16))
    run_name = cfg["model"].get("run_name", cfg["model"].get("baseline", "policy"))
    expert_data_root = args.expert_data_root or Path(cfg.get("data", {}).get("episode_loader", {}).get("root", "data/libero_long"))
    successes = 0
    episodes = 0

    for task_id in task_ids:
        task = suite.get_task(task_id)
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=128, camera_widths=128)
        try:
            env.seed(args.seed + task_id)
            init_states = suite.get_task_init_states(task_id)
            episode_indices = selected_episode_indices(args.split_file, task.name, args.episodes_per_task, len(init_states))
            if args.split_file is not None and not episode_indices:
                print(f"No init states from {args.split_file} matched task {task_id}: {task.name}", flush=True)
            for episode_idx in episode_indices:
                expert_actions = None
                if args.expert_prefix_steps > 0 or args.trace_path is not None:
                    expert_actions = load_expert_actions(expert_data_root, task.name, episode_idx, action_dim)
                obs = env.reset()
                obs = env.set_init_state(init_states[episode_idx])
                history = OnlineHistory(
                    k_recent=k_recent,
                    older_len=older_len,
                    action_dim=action_dim,
                    first_image=obs_to_image(obs, args.camera_key),
                    first_state=obs_to_state(obs),
                )
                success = False
                total_reward = 0.0
                steps = 0
                frames: list[np.ndarray] = []
                if args.video_dir is not None:
                    frames.append(obs_to_image(obs, args.camera_key))
                ensemble_actions: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
                ensemble_raw_actions: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
                ensemble_signal_info: dict[int, list[tuple[int, dict[str, Any], int]]] = defaultdict(list)
                while steps < args.max_steps and not success:
                    controller = "policy"
                    pred_gripper_raw = float("nan")
                    pred_gripper_sign = ""
                    signal_info_for_step: dict[str, Any] = {}
                    signal_offset = 0
                    if expert_actions is not None and steps < args.expert_prefix_steps and steps < len(expert_actions):
                        controller = "expert_prefix"
                        action_sequence = [expert_actions[steps].copy()]
                    else:
                        action_chunk, prediction_info = predict_chunk(
                            model,
                            history,
                            device,
                            args.clip_action,
                            args.discrete_gripper,
                            action_stats,
                            image_normalization,
                            language=task.name,
                            cfg=cfg,
                        )
                        raw_chunk = prediction_info["raw_actions"]
                        if args.temporal_ensemble:
                            for offset, predicted_action in enumerate(action_chunk):
                                ensemble_actions[steps + offset].append((steps, predicted_action.copy()))
                                ensemble_raw_actions[steps + offset].append((steps, raw_chunk[offset].copy()))
                                ensemble_signal_info[steps + offset].append((steps, prediction_info, offset))
                            candidates = ensemble_actions.pop(steps, [(steps, action_chunk[0])])
                            raw_candidates = ensemble_raw_actions.pop(steps, [(steps, raw_chunk[0])])
                            signal_candidates = ensemble_signal_info.pop(steps, [(steps, prediction_info, 0)])
                            weights = np.asarray(
                                [np.exp(-float(args.temporal_ensemble_decay) * max(steps - source_step, 0)) for source_step, _ in candidates],
                                dtype=np.float32,
                            )
                            stacked_actions = np.stack([candidate_action for _, candidate_action in candidates], axis=0)
                            action_sequence = [(stacked_actions * weights[:, None]).sum(axis=0) / weights.sum().clip(min=1e-6)]
                            raw_weights = np.asarray(
                                [np.exp(-float(args.temporal_ensemble_decay) * max(steps - source_step, 0)) for source_step, _ in raw_candidates],
                                dtype=np.float32,
                            )
                            stacked_raw_actions = np.stack([candidate_action for _, candidate_action in raw_candidates], axis=0)
                            raw_action = (stacked_raw_actions * raw_weights[:, None]).sum(axis=0) / raw_weights.sum().clip(min=1e-6)
                            signal_info_for_step = signal_candidates[-1][1]
                            signal_offset = signal_candidates[-1][2]
                            if args.clip_action:
                                action_sequence[0] = np.clip(action_sequence[0], -1.0, 1.0)
                            if args.discrete_gripper and action_sequence[0].shape[-1] >= 7:
                                action_sequence[0][-1] = 1.0 if action_sequence[0][-1] >= 0.0 else -1.0
                            pred_gripper_raw = float(raw_action[-1]) if raw_action.shape[-1] >= 7 else float("nan")
                            pred_gripper_sign = "positive" if pred_gripper_raw >= 0.0 else "negative"
                        else:
                            action_sequence = list(action_chunk[: max(int(args.execute_horizon), 1)])
                            signal_info_for_step = prediction_info
                            signal_offset = 0
                            pred_gripper_raw = float(raw_chunk[0, -1]) if raw_chunk.shape[-1] >= 7 else float("nan")
                            pred_gripper_sign = "positive" if pred_gripper_raw >= 0.0 else "negative"
                    for action in action_sequence:
                        if expert_actions is not None and steps >= args.expert_prefix_steps:
                            controller = "policy"
                        pre_state = obs_to_state(obs)
                        obs, reward, done, _info = env.step(action)
                        post_state = obs_to_state(obs)
                        total_reward += float(reward)
                        steps += 1
                        success = success or bool(reward > 0.0)
                        if args.trace_path is not None:
                            trace_row = {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "suite": args.suite,
                                "run_name": run_name,
                                "seed": args.seed,
                                "task_id": task_id,
                                "task_name": task.name,
                                "episode_idx": episode_idx,
                                "step": steps - 1,
                                "reward": float(reward),
                                "success": int(success),
                                "controller": controller,
                                "expert_prefix_steps": args.expert_prefix_steps,
                                "current_phase_id": int(signal_info_for_step.get("current_phase_id", 0)),
                                "current_secured": int(signal_info_for_step.get("current_secured", 0)),
                                "current_placement_ready": int(signal_info_for_step.get("current_placement_ready", 0)),
                                "target_phase_id": int(np.asarray(signal_info_for_step.get("target_phase_ids", [0]))[min(signal_offset, len(np.asarray(signal_info_for_step.get("target_phase_ids", [0]))) - 1)]),
                                "target_secured": int(np.asarray(signal_info_for_step.get("target_secured", [0]))[min(signal_offset, len(np.asarray(signal_info_for_step.get("target_secured", [0]))) - 1)]),
                                "target_placement_ready": int(np.asarray(signal_info_for_step.get("target_placement_ready", [0]))[min(signal_offset, len(np.asarray(signal_info_for_step.get("target_placement_ready", [0]))) - 1)]),
                                "pred_gripper_raw": pred_gripper_raw,
                                "pred_gripper_sign": pred_gripper_sign,
                                "executed_gripper": float(action[-1]) if len(action) >= 7 else float("nan"),
                                "expert_gripper_if_available": (
                                    float(expert_actions[steps, -1])
                                    if expert_actions is not None and steps < len(expert_actions)
                                    else float("nan")
                                ),
                                "eef_delta_norm": float(np.linalg.norm(post_state[:3] - pre_state[:3])),
                                "action_delta_norm": float(np.linalg.norm(action[:6])),
                            }
                            for dim, value in enumerate(pre_state):
                                trace_row[f"pre_state_{dim}"] = float(value)
                            for dim, value in enumerate(post_state):
                                trace_row[f"post_state_{dim}"] = float(value)
                            for dim, value in enumerate(action):
                                trace_row[f"action_{dim}"] = float(value)
                            append_csv(args.trace_path, trace_row)
                        if args.video_dir is not None and steps % max(args.video_every, 1) == 0:
                            frames.append(obs_to_image(obs, args.camera_key))
                        history.append(obs_to_image(obs, args.camera_key), obs_to_state(obs), action)
                        if success or done or steps >= args.max_steps:
                            break
                video_path = ""
                if args.video_dir is not None:
                    safe_task = task.name.replace("/", "_")
                    video_path = str(
                        args.video_dir
                        / run_name
                        / f"seed{args.seed}_task{task_id:02d}_episode{episode_idx}_{safe_task}.mp4"
                    )
                    write_video(Path(video_path), frames, args.video_fps)
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "suite": args.suite,
                    "run_name": run_name,
                    "checkpoint": str(args.checkpoint),
                    "seed": args.seed,
                    "task_id": task_id,
                    "task_name": task.name,
                    "episode_idx": episode_idx,
                    "success": int(success),
                    "total_reward": total_reward,
                    "steps": steps,
                    "max_steps": args.max_steps,
                    "expert_prefix_steps": args.expert_prefix_steps,
                    "video_path": video_path,
                }
                append_csv(args.results_path, row)
                print(row)
                successes += int(success)
                episodes += 1
        finally:
            env.close()

    success_rate = successes / max(episodes, 1)
    print({"run_name": run_name, "episodes": episodes, "successes": successes, "success_rate": success_rate})


if __name__ == "__main__":
    main()
