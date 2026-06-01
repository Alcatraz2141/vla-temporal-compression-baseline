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

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    h5py = None
    H5PY_IMPORT_ERROR = exc
else:
    H5PY_IMPORT_ERROR = None

try:
    import libero  # noqa: F401
except ImportError as exc:  # pragma: no cover
    LIBERO_IMPORT_ERROR = exc
else:
    LIBERO_IMPORT_ERROR = None


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    source: str
    path: Path
    demo_key: str | None
    length: int


SOURCE_ROOTS = {
    "fractal": Path("data/raw"),
    "ur5": Path("data/raw_diverse"),
    "others": Path("data/raw_diverse"),
    "other": Path("data/raw_diverse"),
}

SOURCE_KEYWORDS = {
    "fractal": ("fractal", "rt-1", "google"),
    "ur5": ("ur5", "berkeley_autolab_ur5"),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pad_left(indices: np.ndarray, target_len: int, default: int = 0) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(indices), dtype=np.bool_)
    if len(indices) >= target_len:
        return indices[-target_len:], mask[-target_len:]
    pad_len = target_len - len(indices)
    return (
        np.concatenate([np.full(pad_len, default, dtype=np.int64), indices.astype(np.int64)]),
        np.concatenate([np.zeros(pad_len, dtype=np.bool_), mask]),
    )


def _take_or_pad(indices: np.ndarray, target_len: int, last_idx: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(indices), dtype=np.bool_)
    if len(indices) >= target_len:
        return indices[:target_len], mask[:target_len]
    pad_len = target_len - len(indices)
    return (
        np.concatenate([indices.astype(np.int64), np.full(pad_len, last_idx, dtype=np.int64)]),
        np.concatenate([mask, np.zeros(pad_len, dtype=np.bool_)]),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_dim_array(values: Any, action_dim: int, default: float) -> np.ndarray:
    if values is None:
        return np.full(action_dim, default, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == action_dim:
        return arr
    out = np.full(action_dim, default, dtype=np.float32)
    out[: min(arr.size, action_dim)] = arr[:action_dim]
    return out


class EpisodeDataset(Dataset):
    """Unified episode dataset for LIBERO HDF5 and local Open X-style episodes."""

    libero_image_key = "agentview_rgb"
    libero_state_key_groups = (
        ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
        ("ee_pos", "ee_ori", "gripper_states"),
        ("ee_states", "gripper_states"),
        ("joint_states", "gripper_states"),
    )

    def __init__(
        self,
        source: str,
        split: str,
        K_recent: int,
        H_action: int,
        max_memory_tokens: int = 16,
        chunk_size: int = 4,
        seed: int = 42,
        root: str | Path | None = None,
        split_dir: str | Path = "splits",
        image_size: int = 128,
        samples_per_epoch: int | None = None,
        eval_windows_per_episode: int = 1,
        max_episodes: int | None = None,
        hdf5_glob: str = "**/*.hdf5",
        stats_path: str | Path | None = None,
        normalize_actions: bool = False,
        action_normalize_dims: list[int] | None = None,
        augment: bool = False,
        image_normalization: str | None = None,
        load_older_context: bool = True,
        task_filter: str | list[str] | None = None,
        transition_sample_prob: float = 0.0,
        transition_sample_radius: int = 3,
    ) -> None:
        self.source = source
        self.split = split
        self.K_recent = int(K_recent)
        self.H_action = int(H_action)
        self.max_older_steps = int(max_memory_tokens) * int(chunk_size)
        self.seed = int(seed)
        self.samples_per_epoch = samples_per_epoch
        self.eval_windows_per_episode = max(int(eval_windows_per_episode), 1)
        self.root = Path(root) if root is not None else self._default_root(source)
        self.split_dir = Path(split_dir)
        self.hdf5_glob = hdf5_glob
        self.image_size = int(image_size)
        self.load_older_context = bool(load_older_context)
        self.task_filter = self._normalize_task_filter(task_filter)
        self.transition_sample_prob = float(transition_sample_prob)
        self.transition_sample_radius = max(int(transition_sample_radius), 0)
        self.action_stats = _read_json(Path(stats_path)) if stats_path else {}
        self.normalize_actions = bool(normalize_actions and self.action_stats)
        self.action_normalize_dims = action_normalize_dims
        self.transform = self._make_transform(image_size, augment, image_normalization)

        all_records = self._discover_records()
        if self.task_filter:
            all_records = [record for record in all_records if self._record_matches_task_filter(record)]
        all_records = [record for record in all_records if record.length >= self.K_recent + self.H_action]
        if not all_records:
            raise FileNotFoundError(
                f"No {source!r} episodes with length >= K_recent + H_action "
                f"({self.K_recent + self.H_action}) under {self.root}"
            )
        self._ensure_splits(all_records)
        wanted = set(self._read_split(self.split_dir / f"{source}_{split}.txt"))
        self.records = [record for record in all_records if record.episode_id in wanted]
        if max_episodes is not None:
            self.records = self.records[: int(max_episodes)]
        if not self.records:
            raise FileNotFoundError(f"No records for source={source}, split={split} in {self.split_dir}")

    def _make_transform(self, image_size: int, augment: bool, image_normalization: str | None) -> transforms.Compose:
        ops: list[Any] = [transforms.ToPILImage()]
        if augment:
            ops.extend(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
                ]
            )
        else:
            ops.append(transforms.Resize((image_size, image_size)))
        ops.append(transforms.ToTensor())
        if image_normalization in {"imagenet", "resnet", "imageNet"}:
            ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        elif image_normalization not in {None, "", "none"}:
            raise ValueError(f"Unsupported image_normalization={image_normalization!r}. Expected 'imagenet' or 'none'.")
        return transforms.Compose(ops)

    def _default_root(self, source: str) -> Path:
        if source == "libero_long":
            return Path("data/libero_long")
        return SOURCE_ROOTS.get(source, Path("data/raw_diverse"))

    def _normalize_task_filter(self, task_filter: str | list[str] | None) -> tuple[str, ...]:
        if task_filter is None:
            return ()
        if isinstance(task_filter, str):
            items = [task_filter]
        else:
            items = list(task_filter)
        return tuple(item.removesuffix("_demo").lower() for item in items if str(item).strip())

    def _record_matches_task_filter(self, record: EpisodeRecord) -> bool:
        if not self.task_filter:
            return True
        task = record.path.stem.removesuffix("_demo").lower()
        return any(wanted in task for wanted in self.task_filter)

    def _discover_records(self) -> list[EpisodeRecord]:
        if self.source == "libero_long":
            return self._discover_libero_records()
        return self._discover_local_records()

    def _discover_local_records(self) -> list[EpisodeRecord]:
        records = []
        for path in sorted(self.root.glob("episode_*")):
            if not path.is_dir():
                continue
            try:
                actions = np.load(path / "actions.npy", mmap_mode="r")
                states = np.load(path / "states.npy", mmap_mode="r")
                image_count = len(list((path / "images").glob("*")))
                length = min(len(actions), len(states), image_count)
            except Exception:
                continue
            metadata = _read_json(path / "metadata.json")
            if not self._matches_source(metadata, path):
                continue
            if length > 0:
                records.append(EpisodeRecord(f"{self.source}::{path.name}", self.source, path, None, length))
        return records

    def _matches_source(self, metadata: dict[str, Any], path: Path) -> bool:
        keywords = SOURCE_KEYWORDS.get(self.source)
        if not keywords:
            return True
        text = " ".join(str(value).lower() for value in metadata.values()) + " " + path.name.lower()
        return any(keyword in text for keyword in keywords)

    def _discover_libero_records(self) -> list[EpisodeRecord]:
        if h5py is None:
            raise ImportError(
                "LIBERO-Long loading needs h5py. Install with `uv add h5py`. "
                "If you also need the official benchmark tools, pip install libero."
            ) from H5PY_IMPORT_ERROR
        records = []
        for file_path in sorted(self.root.glob(self.hdf5_glob)):
            if not file_path.is_file():
                continue
            rel_path = file_path.relative_to(self.root)
            with h5py.File(file_path, "r") as h5:
                demo_keys = self._libero_demo_keys(h5)
                for demo_key in demo_keys:
                    length = self._libero_length(h5[demo_key])
                    if length > 0:
                        records.append(
                            EpisodeRecord(
                                f"{self.source}::{rel_path.as_posix()}::{demo_key}",
                                self.source,
                                file_path,
                                demo_key,
                                length,
                            )
                        )
        return records

    def _libero_demo_keys(self, h5: Any) -> list[str]:
        if "data" in h5:
            return [f"data/{key}" for key in sorted(h5["data"].keys()) if isinstance(h5["data"][key], h5py.Group)]
        return [key for key in sorted(h5.keys()) if isinstance(h5[key], h5py.Group) and "obs" in h5[key]]

    def _libero_length(self, demo: Any) -> int:
        try:
            obs = demo["obs"]
            lengths = [len(demo["actions"]), len(obs[self.libero_image_key])]
            state_keys = self._libero_state_keys(obs)
            lengths.extend(len(obs[key]) for key in state_keys)
            return min(lengths)
        except Exception:
            return 0

    def _libero_state_keys(self, obs: Any) -> tuple[str, ...]:
        for keys in self.libero_state_key_groups:
            if all(key in obs for key in keys):
                return keys
        fallback = tuple(key for key in ("ee_pos", "ee_ori", "ee_states", "gripper_states", "joint_states") if key in obs)
        if fallback:
            return fallback
        raise KeyError(
            "No supported LIBERO proprio keys found. Expected one of: "
            "robot0_eef_pos/robot0_eef_quat/robot0_gripper_qpos, "
            "ee_pos/ee_ori/gripper_states, ee_states/gripper_states, or joint_states/gripper_states."
        )

    def _ensure_splits(self, records: list[EpisodeRecord]) -> None:
        paths = [self.split_dir / f"{self.source}_{name}.txt" for name in ("train", "val", "test")]
        if all(path.exists() for path in paths):
            return
        shuffled = list(records)
        random.Random(self.seed).shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        if n_total >= 3:
            n_val = max(n_val, 1)
            n_train = min(max(n_train, 1), n_total - n_val - 1)
        self._write_split(paths[0], shuffled[:n_train])
        self._write_split(paths[1], shuffled[n_train : n_train + n_val])
        self._write_split(paths[2], shuffled[n_train + n_val :])

    def _read_split(self, path: Path) -> list[str]:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _write_split(self, path: Path, records: list[EpisodeRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(record.episode_id for record in records) + "\n", encoding="utf-8")

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
        min_t = 0
        max_t = length - self.H_action
        if max_t < min_t:
            raise ValueError(f"Episode too short after loading: {record.episode_id}")
        if self.split == "train":
            t = rng.randint(min_t, max_t)
            if self.transition_sample_prob > 0.0 and rng.random() < self.transition_sample_prob:
                action_sign = np.sign(np.asarray(episode["actions"][:, -1], dtype=np.float32))
                transition_idx = np.flatnonzero(action_sign[1:] != action_sign[:-1]) + 1
                if len(transition_idx) > 0:
                    center = int(rng.choice(list(transition_idx)))
                    lo = max(min_t, center - self.transition_sample_radius)
                    hi = min(max_t, center + self.transition_sample_radius)
                    t = rng.randint(lo, hi)
        else:
            t = min_t + round((max_t - min_t) * eval_anchor / max(self.eval_windows_per_episode - 1, 1))

        recent_idx, recent_mask = _pad_left(np.arange(max(0, t - self.K_recent + 1), t + 1), self.K_recent)
        if self.load_older_context and self.max_older_steps > 0:
            older_raw = np.arange(0, max(0, t - self.K_recent + 1))
            if len(older_raw) > self.max_older_steps:
                older_raw = np.linspace(older_raw[0], older_raw[-1], self.max_older_steps).round().astype(np.int64)
            older_idx, older_mask = _pad_left(older_raw, self.max_older_steps)
        else:
            older_idx = np.empty(0, dtype=np.int64)
            older_mask = np.empty(0, dtype=np.bool_)
        target_idx, target_mask = _take_or_pad(np.arange(t, t + self.H_action), self.H_action, length - 1)
        target_transition = np.zeros(self.H_action, dtype=np.bool_)
        for out_i, action_i in enumerate(target_idx):
            if not target_mask[out_i]:
                continue
            prev_i = max(int(action_i) - 1, 0)
            target_transition[out_i] = bool(
                np.sign(episode["actions"][prev_i, -1]) != np.sign(episode["actions"][action_i, -1])
            )
        prev_actions = np.zeros_like(episode["actions"], dtype=np.float32)
        if length > 1:
            prev_actions[1:length] = np.asarray(episode["actions"][: length - 1], dtype=np.float32)
        recent_actions = np.asarray(prev_actions[recent_idx], dtype=np.float32)
        older_actions = np.asarray(prev_actions[older_idx], dtype=np.float32)
        recent_actions[~recent_mask] = 0.0
        older_actions[~older_mask] = 0.0
        target_actions = np.asarray(episode["actions"][target_idx], dtype=np.float32)
        recent_actions = self._normalize_actions(recent_actions)
        older_actions = self._normalize_actions(older_actions)
        target_actions = self._normalize_actions(target_actions)

        return {
            "recent_obs": torch.stack([self._image(episode["images"][i]) for i in recent_idx]),
            "recent_actions": torch.from_numpy(recent_actions),
            "recent_states": torch.from_numpy(np.asarray(episode["states"][recent_idx], dtype=np.float32)),
            "older_obs": (
                torch.stack([self._image(episode["images"][i]) for i in older_idx])
                if len(older_idx) > 0
                else torch.empty((0, 3, self.image_size, self.image_size), dtype=torch.float32)
            ),
            "older_actions": torch.from_numpy(older_actions),
            "older_states": torch.from_numpy(np.asarray(episode["states"][older_idx], dtype=np.float32)),
            "target_actions": torch.from_numpy(target_actions),
            "language": episode["language"],
            "episode_id": record.episode_id,
            "timestep": torch.tensor(t, dtype=torch.long),
            "recent_mask": torch.from_numpy(recent_mask),
            "older_mask": torch.from_numpy(older_mask),
            "target_mask": torch.from_numpy(target_mask),
            "gripper_transition": torch.from_numpy(target_transition),
        }

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if not self.normalize_actions:
            return actions
        stats = self.action_stats.get("actions", self.action_stats)
        action_dim = actions.shape[-1]
        mean = _as_dim_array(stats.get("mean"), action_dim, 0.0)
        std = np.clip(_as_dim_array(stats.get("std"), action_dim, 1.0), 1e-6, None)
        dims = self.action_normalize_dims
        if dims is None:
            dims = stats.get("normalize_dims")
        if dims is None:
            dims = list(range(action_dim))
        valid_dims = [int(dim) for dim in dims if 0 <= int(dim) < action_dim]
        out = actions.astype(np.float32, copy=True)
        if valid_dims:
            out[..., valid_dims] = (out[..., valid_dims] - mean[valid_dims]) / std[valid_dims]
        return out

    def _load_episode(self, record: EpisodeRecord) -> dict[str, Any]:
        if record.source == "libero_long":
            return self._load_libero_episode(record)
        return self._load_local_episode(record)

    def _load_local_episode(self, record: EpisodeRecord) -> dict[str, Any]:
        image_paths = sorted((record.path / "images").glob("*"))
        metadata = _read_json(record.path / "metadata.json")
        return {
            "images": image_paths,
            "actions": np.load(record.path / "actions.npy").astype(np.float32),
            "states": np.load(record.path / "states.npy").astype(np.float32),
            "language": metadata.get("language_instruction", ""),
        }

    def _load_libero_episode(self, record: EpisodeRecord) -> dict[str, Any]:
        with h5py.File(record.path, "r") as h5:
            demo = h5[record.demo_key]
            obs = demo["obs"]
            images = np.asarray(obs[self.libero_image_key])
            state_keys = self._libero_state_keys(obs)
            states = np.concatenate([np.asarray(obs[key]) for key in state_keys], axis=-1).astype(np.float32)
            actions = np.asarray(demo["actions"]).astype(np.float32)
            language = self._language(h5, demo, record)
        length = min(len(images), len(states), len(actions))
        return {"images": images[:length], "states": states[:length], "actions": actions[:length], "language": language}

    def _language(self, h5: Any, demo: Any, record: EpisodeRecord) -> str:
        for source in (demo.attrs, h5.attrs):
            for key in ("language_instruction", "language", "task_description", "instruction"):
                if key in source:
                    value = source[key]
                    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
        return record.path.stem.removesuffix("_demo")

    def _image(self, image: Path | np.ndarray) -> torch.Tensor:
        if isinstance(image, Path):
            arr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if arr is None:
                raise FileNotFoundError(f"Could not read image: {image}")
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            arr = image
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8) if arr.max() <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[0] in {1, 3}:
                arr = np.moveaxis(arr, 0, -1)
        return self.transform(arr)


def episode_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tensor_keys = (
        "recent_obs",
        "recent_actions",
        "recent_states",
        "older_obs",
        "older_actions",
        "older_states",
        "target_actions",
        "timestep",
        "recent_mask",
        "older_mask",
        "target_mask",
        "gripper_transition",
    )
    for key in tensor_keys:
        out[key] = torch.stack([sample[key] for sample in batch])
    out["language"] = [sample["language"] for sample in batch]
    out["episode_id"] = [sample["episode_id"] for sample in batch]
    return out
