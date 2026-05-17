from __future__ import annotations

import io
import json
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.episode_dataset import _pad_indices, _take_or_pad_forward


@dataclass(frozen=True)
class ShardedEpisodeRecord:
    episode_id: str
    shard: Path
    length: int


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class EpisodeShardWindowDataset(Dataset):
    """Episode window sampler backed by episode-level tar shards."""

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
        self.stats = _load_json(Path(stats_path)) if stats_path and Path(stats_path).exists() else {}
        self.transform = self._make_transform(image_size, augment)

        manifest = {}
        with (self.root / "manifest.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                manifest[item["episode_id"]] = item
        split_data = _load_json(self.root / "splits" / f"{split}.json")
        self.records = []
        for episode_id in split_data["episodes"]:
            item = manifest.get(episode_id)
            if not item:
                continue
            shard = self.root / item["shard"]
            length = int(item.get("length", 0))
            if shard.exists() and length >= 2:
                self.records.append(ShardedEpisodeRecord(episode_id, shard, length))
        if not self.records:
            raise FileNotFoundError(f"No valid sharded episodes listed for split={split} under {self.root}")

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

        episode = self._read_episode(record)
        length = min(len(episode["actions"]), len(episode["states"]), len(episode["images"]))
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

        return {
            "recent_obs": torch.stack([self._decode_image(episode["images"][i]) for i in recent_idx]),
            "recent_actions": self._actions(episode["actions"], recent_idx),
            "recent_states": torch.from_numpy(np.asarray(episode["states"][recent_idx], dtype=np.float32)),
            "recent_mask": torch.from_numpy(recent_mask),
            "older_obs": torch.stack([self._decode_image(episode["images"][i]) for i in older_idx]),
            "older_actions": self._actions(episode["actions"], older_idx),
            "older_states": torch.from_numpy(np.asarray(episode["states"][older_idx], dtype=np.float32)),
            "older_mask": torch.from_numpy(older_mask),
            "target_actions": self._actions(episode["actions"], target_idx),
            "target_mask": torch.from_numpy(target_mask),
            "language": episode["metadata"].get("language_instruction", ""),
            "episode_id": record.episode_id,
            "timestep": torch.tensor(t, dtype=torch.long),
        }

    def _read_episode(self, record: ShardedEpisodeRecord) -> dict[str, Any]:
        prefix = f"{record.episode_id}/"
        with tarfile.open(record.shard, "r") as tar:
            actions = np.load(io.BytesIO(tar.extractfile(prefix + "actions.npy").read()))
            states = np.load(io.BytesIO(tar.extractfile(prefix + "states.npy").read()))
            metadata = json.loads(tar.extractfile(prefix + "metadata.json").read().decode("utf-8"))
            image_members = sorted(name for name in tar.getnames() if name.startswith(prefix + "images/"))
            images = [tar.extractfile(name).read() for name in image_members]
        if min(len(actions), len(states), len(images)) < 2:
            raise ValueError(f"Episode too short: {record.episode_id}")
        return {"actions": actions, "states": states, "metadata": metadata, "images": images}

    def _actions(self, actions: np.ndarray, idx: np.ndarray) -> torch.Tensor:
        out = torch.from_numpy(np.asarray(actions[idx], dtype=np.float32))
        stats = self.stats.get("actions", {})
        if "mean" in stats and "std" in stats:
            mean = torch.tensor(stats["mean"], dtype=out.dtype)
            std = torch.tensor(stats["std"], dtype=out.dtype).clamp_min(1e-6)
            out = (out - mean) / std
        return out

    def _decode_image(self, image_bytes: bytes) -> torch.Tensor:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from shard")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image)
