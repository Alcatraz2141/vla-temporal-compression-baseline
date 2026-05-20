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
        self.transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )

        all_records = self._discover_records()
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

    def _default_root(self, source: str) -> Path:
        if source == "libero_long":
            return Path("data/libero_long")
        return SOURCE_ROOTS.get(source, Path("data/raw_diverse"))

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
        else:
            t = min_t + round((max_t - min_t) * eval_anchor / max(self.eval_windows_per_episode - 1, 1))

        recent_idx, recent_mask = _pad_left(np.arange(max(0, t - self.K_recent + 1), t + 1), self.K_recent)
        older_raw = np.arange(0, max(0, t - self.K_recent + 1))
        if len(older_raw) > self.max_older_steps:
            older_raw = np.linspace(older_raw[0], older_raw[-1], self.max_older_steps).round().astype(np.int64)
        older_idx, older_mask = _pad_left(older_raw, self.max_older_steps)
        target_idx, target_mask = _take_or_pad(np.arange(t, t + self.H_action), self.H_action, length - 1)
        prev_actions = np.zeros_like(episode["actions"], dtype=np.float32)
        if length > 1:
            prev_actions[1:length] = np.asarray(episode["actions"][: length - 1], dtype=np.float32)
        recent_actions = np.asarray(prev_actions[recent_idx], dtype=np.float32)
        older_actions = np.asarray(prev_actions[older_idx], dtype=np.float32)
        recent_actions[~recent_mask] = 0.0
        older_actions[~older_mask] = 0.0

        return {
            "recent_obs": torch.stack([self._image(episode["images"][i]) for i in recent_idx]),
            "recent_actions": torch.from_numpy(recent_actions),
            "recent_states": torch.from_numpy(np.asarray(episode["states"][recent_idx], dtype=np.float32)),
            "older_obs": torch.stack([self._image(episode["images"][i]) for i in older_idx]),
            "older_actions": torch.from_numpy(older_actions),
            "older_states": torch.from_numpy(np.asarray(episode["states"][older_idx], dtype=np.float32)),
            "target_actions": torch.from_numpy(np.asarray(episode["actions"][target_idx], dtype=np.float32)),
            "language": episode["language"],
            "episode_id": record.episode_id,
            "timestep": torch.tensor(t, dtype=torch.long),
            "recent_mask": torch.from_numpy(recent_mask),
            "older_mask": torch.from_numpy(older_mask),
            "target_mask": torch.from_numpy(target_mask),
        }

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
    )
    for key in tensor_keys:
        out[key] = torch.stack([sample[key] for sample in batch])
    out["language"] = [sample["language"] for sample in batch]
    out["episode_id"] = [sample["episode_id"] for sample in batch]
    return out
