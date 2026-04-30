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
class SequenceIndex:
    episode_dir: Path
    start: int
    length: int


def _load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class VLADataset(Dataset):
    """Lazy dataset over preprocessed local VLA episodes.

    Expected episode layout:
      episode_xxxxxx/
        images/000000.jpg
        actions.npy
        states.npy
        metadata.json
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        T_obs: int = 4,
        T_action: int = 16,
        image_size: int = 224,
        baseline: str = "sliding_window",
        augment: bool = False,
        stats_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.T_obs = 1 if baseline == "no_temporal" else int(T_obs)
        self.T_action = int(T_action)
        self.image_size = int(image_size)
        self.baseline = baseline
        self.augment = augment
        self.stats = _load_json(Path(stats_path), {}) if stats_path else {}

        split_root = self.root / split
        self.episode_dirs = sorted(split_root.glob("episode_*")) if split_root.exists() else sorted(self.root.glob("episode_*"))
        self.index = self._build_index()
        if not self.index:
            raise FileNotFoundError(
                f"No sequences found under {self.root}. Run datasets/preprocess.py or scripts/download_data.py first."
            )

        self.transform = self._make_transform()

    def _build_index(self) -> list[SequenceIndex]:
        index: list[SequenceIndex] = []
        stride = 1
        for episode_dir in self.episode_dirs:
            actions_path = episode_dir / "actions.npy"
            if not actions_path.exists():
                continue
            length = int(np.load(actions_path, mmap_mode="r").shape[0])
            max_start = max(0, length - max(self.T_obs, self.T_action))
            for start in range(0, max_start + 1, stride):
                index.append(SequenceIndex(episode_dir=episode_dir, start=start, length=length))
        return index

    def _make_transform(self) -> transforms.Compose:
        ops: list[Any] = [transforms.ToPILImage()]
        if self.augment and self.split == "train":
            ops.extend(
                [
                    transforms.RandomResizedCrop(self.image_size, scale=(0.85, 1.0)),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
                ]
            )
        else:
            ops.extend([transforms.Resize(self.image_size), transforms.CenterCrop(self.image_size)])
        ops.append(transforms.ToTensor())
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.index[idx]
        actions = np.load(item.episode_dir / "actions.npy", mmap_mode="r")
        states = np.load(item.episode_dir / "states.npy", mmap_mode="r")
        metadata = _load_json(item.episode_dir / "metadata.json", {})
        image_paths = sorted((item.episode_dir / "images").glob("*"))

        obs_indices = self._window_indices(item.start, self.T_obs, item.length)
        action_indices = self._window_indices(item.start, self.T_action, item.length)

        images = torch.stack([self._load_image(image_paths[i]) for i in obs_indices])
        states_t = torch.from_numpy(np.asarray(states[obs_indices], dtype=np.float32))
        actions_t = torch.from_numpy(np.asarray(actions[action_indices], dtype=np.float32))

        actions_t = self._normalize_actions(actions_t)
        mask = torch.ones(self.T_action, dtype=torch.bool)
        return {
            "images": images,
            "states": states_t,
            "actions": actions_t,
            "mask": mask,
            "language": metadata.get("language_instruction", ""),
            "episode": item.episode_dir.name,
        }

    @staticmethod
    def _window_indices(start: int, size: int, episode_len: int) -> np.ndarray:
        raw = np.arange(start, start + size)
        return np.clip(raw, 0, max(episode_len - 1, 0)).astype(np.int64)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image)

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        action_stats = self.stats.get("actions", {})
        mean = action_stats.get("mean")
        std = action_stats.get("std")
        if mean is None or std is None:
            return actions
        mean_t = torch.tensor(mean, dtype=actions.dtype)
        std_t = torch.tensor(std, dtype=actions.dtype).clamp_min(1e-6)
        return (actions - mean_t) / std_t


def vla_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("images", "states", "actions", "mask"):
        out[key] = torch.stack([sample[key] for sample in batch])
    out["language"] = [sample.get("language", "") for sample in batch]
    out["episode"] = [sample.get("episode", "") for sample in batch]
    return out


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
