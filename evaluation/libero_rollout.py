from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import deque
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
from utils.seed import resolve_device, set_seed


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


def image_tensor(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.stack(images, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)
    return tensor.to(device)


def vector_tensor(vectors: list[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack(vectors, axis=0).astype(np.float32)).unsqueeze(0).to(device)


def mask_tensor(mask: list[bool], device: torch.device) -> torch.Tensor:
    return torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)


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
) -> np.ndarray:
    recent_images, recent_states, recent_actions, recent_mask = history.recent()
    if isinstance(model, EventGatedMemoryVLA):
        older_images, older_states, older_actions, older_mask = history.older()
        pred = model(
            recent_obs=image_tensor(recent_images, device),
            recent_actions=vector_tensor(recent_actions, device),
            recent_states=vector_tensor(recent_states, device),
            older_obs=image_tensor(older_images, device),
            older_actions=vector_tensor(older_actions, device),
            older_states=vector_tensor(older_states, device),
            recent_mask=mask_tensor(recent_mask, device),
            older_mask=mask_tensor(older_mask, device),
        )
    else:
        pred = model(images=image_tensor(recent_images, device), states=vector_tensor(recent_states, device))
    actions = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if clip_action:
        actions = np.clip(actions, -1.0, 1.0)
    return actions


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_policy(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], int, int]:
    cfg = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_cfg = checkpoint.get("config", cfg)
    ckpt_cfg["model"]["baseline"] = cfg["model"].get("baseline", ckpt_cfg["model"].get("baseline", "sliding_window"))
    ckpt_cfg["model"]["run_name"] = cfg["model"].get("run_name", ckpt_cfg["model"].get("run_name", ckpt_cfg["model"]["baseline"]))
    model = build_model(ckpt_cfg, int(checkpoint["state_dim"]), int(checkpoint["action_dim"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, ckpt_cfg, int(checkpoint["state_dim"]), int(checkpoint["action_dim"])


def parse_tasks(value: str | None, total: int) -> list[int]:
    if value is None or value == "all":
        return list(range(total))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained PyTorch policies in the official LIBERO simulator.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--tasks", default="0", help="Comma-separated task ids, or 'all'.")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--camera-key", default="agentview_image")
    parser.add_argument("--results-path", type=Path, default=Path("results/libero_rollouts.csv"))
    parser.add_argument("--clip-action", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    set_seed(args.seed)
    device = resolve_device(args.device)
    benchmark, get_libero_path, OffScreenRenderEnv = _import_libero()
    model, cfg, state_dim, action_dim = load_policy(args.config, args.checkpoint, device)
    if state_dim != 8 or action_dim != 7:
        raise ValueError(f"Expected LIBERO state_dim=8 and action_dim=7, got state_dim={state_dim}, action_dim={action_dim}.")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = parse_tasks(args.tasks, suite.n_tasks)
    k_recent = int(cfg["data"].get("K_recent", cfg["data"].get("T_obs", 8)))
    older_len = int(cfg.get("memory", {}).get("chunk_size", 4)) * int(cfg.get("memory", {}).get("max_memory_tokens", 16))
    run_name = cfg["model"].get("run_name", cfg["model"].get("baseline", "policy"))
    successes = 0
    episodes = 0

    for task_id in task_ids:
        task = suite.get_task(task_id)
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=128, camera_widths=128)
        try:
            env.seed(args.seed + task_id)
            init_states = suite.get_task_init_states(task_id)
            for episode_idx in range(min(args.episodes_per_task, len(init_states))):
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
                while steps < args.max_steps and not success:
                    action_chunk = predict_chunk(model, history, device, args.clip_action)
                    for action in action_chunk:
                        obs, reward, done, _info = env.step(action)
                        total_reward += float(reward)
                        steps += 1
                        success = success or bool(reward > 0.0)
                        history.append(obs_to_image(obs, args.camera_key), obs_to_state(obs), action)
                        if success or done or steps >= args.max_steps:
                            break
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
