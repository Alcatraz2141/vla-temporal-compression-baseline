from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.episode_dataset import _pad_indices, _take_or_pad_forward

try:
    import h5py
except ImportError as exc:  # pragma: no cover - depends on local env
    h5py = None
    H5PY_IMPORT_ERROR = exc
else:
    H5PY_IMPORT_ERROR = None

try:
    import libero  # noqa: F401
except ImportError as exc:  # pragma: no cover - LIBERO is optional
    LIBERO_IMPORT_ERROR = exc
else:
    LIBERO_IMPORT_ERROR = None


@dataclass(frozen=True)
class LiberoEpisodeRecord:
    episode_id: str
    file_path: Path
    demo_key: str
    length: int


def _read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_split(path: Path, records: list[LiberoEpisodeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(record.episode_id for record in records) + "\n", encoding="utf-8")


def _decode_episode_id(episode_id: str) -> tuple[Path, str]:
    rel, demo_key = episode_id.split("::", maxsplit=1)
    return Path(rel), demo_key


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError(
            "LIBERO-Long HDF5 loading requires h5py. Install it with `uv add h5py`. "
            "The optional `libero` package is not required for direct HDF5 loading."
        ) from H5PY_IMPORT_ERROR


class LiberoLongDataset(Dataset):
    """LIBERO-Long HDF5 episode sampler with in-dataloader window construction."""

    image_key = "agentview_rgb"
    state_key_groups = (
        ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
        ("ee_pos", "ee_ori", "gripper_states"),
        ("ee_states", "gripper_states"),
        ("joint_states", "gripper_states"),
    )

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
        hdf5_glob: str = "**/*.hdf5",
        split_dir: str | Path | None = None,
        samples_per_epoch: int | None = None,
        eval_windows_per_episode: int = 1,
        max_episodes: int | None = None,
    ) -> None:
        _require_h5py()
        self.root = Path(root)
        self.split = split
        self.K_recent = int(K_recent)
        self.H_action = int(H_action)
        self.max_older_steps = int(max_older_steps)
        self.seed = int(seed)
        self.samples_per_epoch = samples_per_epoch
        self.eval_windows_per_episode = max(int(eval_windows_per_episode), 1)
        self.transform = self._make_transform(image_size, augment)
        self.split_dir = Path(split_dir) if split_dir else self.root / "splits"

        all_records = self._discover_records(hdf5_glob)
        if not all_records:
            raise FileNotFoundError(
                f"No valid LIBERO HDF5 episodes found under {self.root} with glob {hdf5_glob!r}. "
                "Expected demos with obs/agentview_rgb, actions, and robot proprio keys."
            )
        self._ensure_splits(all_records)
        split_path = self.split_dir / f"libero_long_{split}.txt"
        wanted = set(_read_split(split_path))
        self.records = [record for record in all_records if record.episode_id in wanted]
        if max_episodes is not None:
            self.records = self.records[: int(max_episodes)]
        if not self.records:
            raise FileNotFoundError(f"No LIBERO-Long episodes listed for split={split} in {split_path}")

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

    def _hdf5_files(self, hdf5_glob: str) -> list[Path]:
        if any(ch in hdf5_glob for ch in "*?[]"):
            return sorted(path for path in self.root.glob(hdf5_glob) if path.is_file())
        path = self.root / hdf5_glob
        return [path] if path.exists() else []

    def _discover_records(self, hdf5_glob: str) -> list[LiberoEpisodeRecord]:
        records: list[LiberoEpisodeRecord] = []
        for file_path in self._hdf5_files(hdf5_glob):
            rel_path = file_path.relative_to(self.root)
            with h5py.File(file_path, "r") as h5:
                demo_keys = self._demo_keys(h5)
                for demo_key in demo_keys:
                    length = self._episode_length(h5[demo_key])
                    if length >= 2:
                        episode_id = f"{rel_path.as_posix()}::{demo_key}"
                        records.append(LiberoEpisodeRecord(episode_id, file_path, demo_key, length))
        return sorted(records, key=lambda record: record.episode_id)

    def _demo_keys(self, h5: Any) -> list[str]:
        if "data" in h5:
            return [f"data/{key}" for key in sorted(h5["data"].keys()) if isinstance(h5["data"][key], h5py.Group)]
        return [key for key in sorted(h5.keys()) if isinstance(h5[key], h5py.Group) and "obs" in h5[key]]

    def _episode_length(self, demo: Any) -> int:
        try:
            obs = demo["obs"]
            lengths = [len(obs[self.image_key]), len(demo["actions"])]
            lengths.extend(len(obs[key]) for key in self._state_keys(obs))
            return min(lengths)
        except Exception:
            return 0

    def _state_keys(self, obs: Any) -> tuple[str, ...]:
        for keys in self.state_key_groups:
            if all(key in obs for key in keys):
                return keys
        fallback = tuple(key for key in ("ee_pos", "ee_ori", "ee_states", "gripper_states", "joint_states") if key in obs)
        if fallback:
            return fallback
        raise KeyError("No supported LIBERO proprio keys found.")

    def _ensure_splits(self, records: list[LiberoEpisodeRecord]) -> None:
        train_path = self.split_dir / "libero_long_train.txt"
        val_path = self.split_dir / "libero_long_val.txt"
        test_path = self.split_dir / "libero_long_test.txt"
        if train_path.exists() and val_path.exists() and test_path.exists():
            return
        shuffled = list(records)
        random.Random(self.seed).shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        if n_total >= 3:
            n_val = max(n_val, 1)
            n_train = min(max(n_train, 1), n_total - n_val - 1)
        _write_split(train_path, shuffled[:n_train])
        _write_split(val_path, shuffled[n_train : n_train + n_val])
        _write_split(test_path, shuffled[n_train + n_val :])

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

        episode = self._load_episode(record)
        length = min(len(episode["images"]), len(episode["actions"]), len(episode["states"]))
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
            "episode_id": record.episode_id,
            "recent_obs": torch.stack([self._image(episode["images"][i]) for i in recent_idx]),
            "recent_actions": torch.from_numpy(np.asarray(episode["actions"][recent_idx], dtype=np.float32)),
            "recent_states": torch.from_numpy(np.asarray(episode["states"][recent_idx], dtype=np.float32)),
            "recent_mask": torch.from_numpy(recent_mask),
            "older_obs": torch.stack([self._image(episode["images"][i]) for i in older_idx]),
            "older_actions": torch.from_numpy(np.asarray(episode["actions"][older_idx], dtype=np.float32)),
            "older_states": torch.from_numpy(np.asarray(episode["states"][older_idx], dtype=np.float32)),
            "older_mask": torch.from_numpy(older_mask),
            "target_actions": torch.from_numpy(np.asarray(episode["actions"][target_idx], dtype=np.float32)),
            "target_mask": torch.from_numpy(target_mask),
            "language": episode["language"],
            "timestep": torch.tensor(t, dtype=torch.long),
        }

    def _load_episode(self, record: LiberoEpisodeRecord) -> dict[str, Any]:
        with h5py.File(record.file_path, "r") as h5:
            demo = h5[record.demo_key]
            obs = demo["obs"]
            images = np.asarray(obs[self.image_key])
            states = np.concatenate([np.asarray(obs[key]) for key in self._state_keys(obs)], axis=-1)
            actions = np.asarray(demo["actions"])
            language = self._language(h5, demo, record)
        length = min(len(images), len(states), len(actions))
        return {
            "images": images[:length],
            "states": states[:length].astype(np.float32),
            "actions": actions[:length].astype(np.float32),
            "language": language,
        }

    def _language(self, h5: Any, demo: Any, record: LiberoEpisodeRecord) -> str:
        for source in (demo.attrs, h5.attrs):
            for key in ("language_instruction", "language", "task_description", "instruction"):
                if key in source:
                    value = source[key]
                    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
        if "env_args" in h5.attrs:
            try:
                env_args = json.loads(h5.attrs["env_args"])
                for key in ("language_instruction", "task_description", "problem_name", "task"):
                    if key in env_args:
                        return str(env_args[key])
            except Exception:
                pass
        return record.file_path.stem

    def _image(self, image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255 if image.max() > 1.0 else 1.0)
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        if image.ndim == 3 and image.shape[0] in {1, 3}:
            image = np.moveaxis(image, 0, -1)
        return self.transform(image)
