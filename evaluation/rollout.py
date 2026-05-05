from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.vla_baseline import build_model
from utils.config import load_config
from utils.seed import resolve_device, set_seed


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pad_truncate(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size >= dim:
        return x[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[: x.size] = x
    return out


def _as_image(obs: Any, env: Any, image_key: str | None) -> np.ndarray:
    image = None
    if isinstance(obs, dict) and image_key:
        image = obs.get(image_key)
    if image is None and isinstance(obs, dict):
        for key in ("image", "rgb", "rgb_static", "agentview_image", "pixels"):
            if key in obs:
                image = obs[key]
                break
    if image is None and hasattr(env, "render"):
        image = env.render()
    if image is None:
        raise ValueError("Could not find an RGB observation. Set rollout.image_key or use an env with render().")
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.dtype in (np.float32, np.float64):
        image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype(np.uint8)


def _as_state(obs: Any, state_key: str | None, state_dim: int) -> np.ndarray:
    if isinstance(obs, dict):
        if state_key and state_key in obs:
            return _pad_truncate(obs[state_key], state_dim)
        parts = []
        for key in ("state", "proprio", "robot_state", "joint_positions", "tcp_pose"):
            if key in obs:
                parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
        if parts:
            return _pad_truncate(np.concatenate(parts), state_dim)
    return np.zeros(state_dim, dtype=np.float32)


def _success(info: dict[str, Any], reward: float, cfg: dict[str, Any]) -> bool:
    success_key = cfg.get("success_key", "success")
    if success_key in info:
        return bool(info[success_key])
    threshold = cfg.get("reward_success_threshold")
    return bool(threshold is not None and reward >= float(threshold))


class RolloutPolicy:
    def __init__(self, model: torch.nn.Module, cfg: dict[str, Any], device: torch.device) -> None:
        self.model = model.eval()
        self.cfg = cfg
        self.device = device
        self.T_obs = int(cfg["data"]["T_obs"])
        if cfg["model"].get("baseline") == "no_temporal":
            self.T_obs = 1
        if cfg["model"].get("baseline") == "larger_window":
            self.T_obs *= 2
        self.state_dim = int(cfg["data"]["state_dim"])
        self.action_dim = int(cfg["data"]["action_dim"])
        self.image_key = cfg["rollout"].get("image_key")
        self.state_key = cfg["rollout"].get("state_key")
        self.stats = _load_json(cfg["data"].get("normalization", {}).get("stats_path"))
        self.image_tf = transforms.Compose(
            [
                transforms.Resize(int(cfg["data"].get("image_size", 224))),
                transforms.CenterCrop(int(cfg["data"].get("image_size", 224))),
                transforms.ToTensor(),
            ]
        )
        self.images: deque[torch.Tensor] = deque(maxlen=self.T_obs)
        self.states: deque[torch.Tensor] = deque(maxlen=self.T_obs)

    def reset(self, obs: Any, env: Any) -> None:
        self.images.clear()
        self.states.clear()
        image, state = self._encode_obs(obs, env)
        for _ in range(self.T_obs):
            self.images.append(image)
            self.states.append(state)

    def _encode_obs(self, obs: Any, env: Any) -> tuple[torch.Tensor, torch.Tensor]:
        image = _as_image(obs, env, self.image_key)
        state = _as_state(obs, self.state_key, self.state_dim)
        return self.image_tf(Image.fromarray(image)), torch.from_numpy(state)

    def observe(self, obs: Any, env: Any) -> None:
        image, state = self._encode_obs(obs, env)
        self.images.append(image)
        self.states.append(state)

    def act_chunk(self) -> np.ndarray:
        images = torch.stack(list(self.images)).unsqueeze(0).to(self.device)
        states = torch.stack(list(self.states)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(images=images, states=states).squeeze(0).detach().cpu()
        actions = self._unnormalize_actions(pred).numpy()
        return actions[:, : self.action_dim]

    def _unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        stats = self.stats.get("actions", {})
        if "mean" not in stats or "std" not in stats:
            return actions
        mean = torch.tensor(stats["mean"], dtype=actions.dtype)
        std = torch.tensor(stats["std"], dtype=actions.dtype)
        return actions * std + mean


def make_env(env_id: str, render_mode: str | None) -> Any:
    import gymnasium as gym

    kwargs = {}
    if render_mode:
        kwargs["render_mode"] = render_mode
    try:
        return gym.make(env_id, **kwargs)
    except TypeError:
        return gym.make(env_id)


def adapt_action(env: Any, action: np.ndarray) -> Any:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    space = getattr(env, "action_space", None)
    if hasattr(space, "n"):
        return int(np.clip(round(float(action[0])), 0, int(space.n) - 1))
    shape = getattr(space, "shape", None)
    if shape is not None and len(shape) > 0:
        target_dim = int(np.prod(shape))
        action = _pad_truncate(action, target_dim).reshape(shape)
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is not None and high is not None:
        action = np.clip(action, low, high)
    return action


def run_rollouts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rollout_cfg = cfg["rollout"]
    if not rollout_cfg.get("env_id"):
        raise ValueError("Set rollout.env_id to a Gymnasium-compatible simulator env id.")
    device = resolve_device(cfg.get("device", "auto"))
    checkpoint = torch.load(rollout_cfg["checkpoint"], map_location=device)
    ckpt_cfg = checkpoint.get("config", cfg)
    ckpt_cfg["data"].update({k: v for k, v in cfg["data"].items() if v is not None})
    ckpt_cfg["rollout"] = rollout_cfg
    model = build_model(ckpt_cfg, int(checkpoint["state_dim"]), int(checkpoint["action_dim"])).to(device)
    model.load_state_dict(checkpoint["model"])
    policy = RolloutPolicy(model, ckpt_cfg, device)
    env = make_env(rollout_cfg["env_id"], rollout_cfg.get("render_mode"))

    rows: list[dict[str, Any]] = []
    for ep in range(int(rollout_cfg.get("episodes", 20))):
        obs, _ = env.reset(seed=int(cfg.get("seed", 42)) + ep)
        policy.reset(obs, env)
        total_reward = 0.0
        success = False
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated) and steps < int(rollout_cfg.get("max_steps", 300)):
            actions = policy.act_chunk()
            for action in actions:
                env_action = adapt_action(env, action)
                for _ in range(int(rollout_cfg.get("action_repeat", 1))):
                    obs, reward, terminated, truncated, info = env.step(env_action)
                    total_reward += float(reward)
                    steps += 1
                    success = success or _success(info, float(reward), rollout_cfg)
                    if terminated or truncated or steps >= int(rollout_cfg.get("max_steps", 300)):
                        break
                    policy.observe(obs, env)
                if terminated or truncated or steps >= int(rollout_cfg.get("max_steps", 300)):
                    break
        rows.append({"episode": ep, "success": float(success), "return": total_reward, "steps": steps})
    env.close()
    return rows


def append_summary(path: Path, cfg: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env_id": cfg["rollout"]["env_id"],
        "baseline": cfg["model"].get("baseline", "sliding_window"),
        "checkpoint": cfg["rollout"]["checkpoint"],
        "seed": cfg.get("seed", 42),
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])) if rows else float("nan"),
        "mean_return": float(np.mean([r["return"] for r in rows])) if rows else float("nan"),
        "mean_steps": float(np.mean([r["steps"] for r in rows])) if rows else float("nan"),
    }
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(summary)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulator rollouts for a trained VLA checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/sim_rollout.yaml"))
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.env_id is not None:
        cfg["rollout"]["env_id"] = args.env_id
    if args.checkpoint is not None:
        cfg["rollout"]["checkpoint"] = str(args.checkpoint)
    if args.episodes is not None:
        cfg["rollout"]["episodes"] = args.episodes
    set_seed(int(cfg.get("seed", 42)))
    rows = run_rollouts(cfg)
    append_summary(Path(cfg["rollout"].get("results_path", "results/rollouts.csv")), cfg, rows)


if __name__ == "__main__":
    main()
