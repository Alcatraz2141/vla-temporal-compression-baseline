from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    path: Path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pad_indices(indices: np.ndarray, target_len: int, default: int = 0) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(indices), dtype=np.bool_)
    if len(indices) >= target_len:
        return indices[-target_len:], mask[-target_len:]
    pad_len = target_len - len(indices)
    padded = np.concatenate([np.full(pad_len, default, dtype=np.int64), indices.astype(np.int64)])
    mask = np.concatenate([np.zeros(pad_len, dtype=np.bool_), mask])
    return padded, mask


def _take_or_pad_forward(indices: np.ndarray, target_len: int, last_idx: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(indices), dtype=np.bool_)
    if len(indices) >= target_len:
        return indices[:target_len], mask[:target_len]
    pad_len = target_len - len(indices)
    padded = np.concatenate([indices.astype(np.int64), np.full(pad_len, last_idx, dtype=np.int64)])
    mask = np.concatenate([mask, np.zeros(pad_len, dtype=np.bool_)])
    return padded, mask


class EpisodeWindowDataset(Dataset):
    """Episode-level sampler that constructs windows and older context on demand."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        K_recent: int,
        H_action: int,
        image_size: int = 224,
        max_older_steps: int = 64,
        seed: int = 42,
        augment: bool = False,
        stats_path: str | Path | None = None,
        samples_per_epoch: int | None = None,
        eval_windows_per_episode: int = 1,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.K_recent = int(K_recent)
        self.H_action = int(H_action)
        self.max_older_steps = int(max_older_steps)
        self.seed = int(seed)
        self.samples_per_epoch = samples_per_epoch
        self.eval_windows_per_episode = max(int(eval_windows_per_episode), 1)
        split_data = _load_json(self.root / "splits" / f"{split}.json")
        self.records = [
            EpisodeRecord(eid, self.root / "episodes" / eid)
            for eid in split_data["episodes"]
            if self._is_valid_episode(self.root / "episodes" / eid)
        ]
        if not self.records:
            raise FileNotFoundError(f"No episodes listed for split={split} under {self.root}/splits")
        self.stats = _load_json(Path(stats_path)) if stats_path and Path(stats_path).exists() else {}
        self.transform = self._make_transform(image_size, augment)

    def _is_valid_episode(self, path: Path) -> bool:
        try:
            actions = np.load(path / "actions.npy", mmap_mode="r")
            states = np.load(path / "states.npy", mmap_mode="r")
            image_count = len(list((path / "images").glob("*")))
            return min(len(actions), len(states), image_count) >= 2
        except Exception:
            return False

    def _make_transform(self, image_size: int, augment: bool) -> transforms.Compose:
        ops: list[Any] = [transforms.ToPILImage()]
        if augment:
            ops.extend(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
                ]
            )
        else:
            ops.extend([transforms.Resize(image_size), transforms.CenterCrop(image_size)])
        ops.append(transforms.ToTensor())
        return transforms.Compose(ops)

    def __len__(self) -> int:
        if self.split == "train":
            return self.samples_per_epoch or len(self.records)
        return len(self.records) * self.eval_windows_per_episode

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = random.Random(self.seed + idx)
        if self.split == "train":
            record = rng.choice(self.records)
            eval_anchor = 0
        else:
            record = self.records[(idx // self.eval_windows_per_episode) % len(self.records)]
            eval_anchor = idx % self.eval_windows_per_episode
        actions = np.load(record.path / "actions.npy", mmap_mode="r")
        states = np.load(record.path / "states.npy", mmap_mode="r")
        image_paths = sorted((record.path / "images").glob("*"))
        length = min(len(actions), len(states), len(image_paths))
        if length < 2:
            raise ValueError(f"Episode too short: {record.path}")
        min_t = min(self.K_recent, length - 1)
        max_t = max(min_t, length - 1)
        if self.split == "train":
            t = rng.randint(min_t, max_t)
        else:
            span = max(max_t - min_t, 0)
            t = min_t + round(span * eval_anchor / max(self.eval_windows_per_episode - 1, 1))
        t = min(t, length - 1)

        recent_raw = np.arange(max(0, t - self.K_recent), t)
        recent_idx, recent_mask = _pad_indices(recent_raw, self.K_recent, default=0)
        older_raw = np.arange(0, max(0, t - self.K_recent))
        if len(older_raw) > self.max_older_steps:
            older_raw = np.linspace(older_raw[0], older_raw[-1], self.max_older_steps).round().astype(np.int64)
        older_idx, older_mask = _pad_indices(older_raw, self.max_older_steps, default=0)
        target_raw = np.arange(t, min(length, t + self.H_action))
        target_idx, target_mask = _take_or_pad_forward(target_raw, self.H_action, last_idx=length - 1)

        metadata = _load_json(record.path / "metadata.json")
        return {
            "recent_obs": torch.stack([self._load_image(image_paths[i]) for i in recent_idx]),
            "recent_actions": self._actions(actions, recent_idx),
            "recent_states": torch.from_numpy(np.asarray(states[recent_idx], dtype=np.float32)),
            "recent_mask": torch.from_numpy(recent_mask),
            "older_obs": torch.stack([self._load_image(image_paths[i]) for i in older_idx]),
            "older_actions": self._actions(actions, older_idx),
            "older_states": torch.from_numpy(np.asarray(states[older_idx], dtype=np.float32)),
            "older_mask": torch.from_numpy(older_mask),
            "target_actions": self._actions(actions, target_idx),
            "target_mask": torch.from_numpy(target_mask),
            "language": metadata.get("language_instruction", ""),
            "episode_id": record.episode_id,
            "timestep": torch.tensor(t, dtype=torch.long),
        }

    def _actions(self, actions: np.ndarray, idx: np.ndarray) -> torch.Tensor:
        out = torch.from_numpy(np.asarray(actions[idx], dtype=np.float32))
        stats = self.stats.get("actions", {})
        if "mean" in stats and "std" in stats:
            mean = torch.tensor(stats["mean"], dtype=out.dtype)
            std = torch.tensor(stats["std"], dtype=out.dtype).clamp_min(1e-6)
            out = (out - mean) / std
        return out

    def _load_image(self, image_path: Path) -> torch.Tensor:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image)


def episode_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tensor_keys = (
        "recent_obs",
        "recent_actions",
        "recent_states",
        "recent_mask",
        "older_obs",
        "older_actions",
        "older_states",
        "older_mask",
        "target_actions",
        "target_mask",
        "timestep",
    )
    for key in tensor_keys:
        out[key] = torch.stack([sample[key] for sample in batch])
    out["language"] = [sample["language"] for sample in batch]
    out["episode_id"] = [sample["episode_id"] for sample in batch]
    return out
