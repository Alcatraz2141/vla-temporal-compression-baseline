from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torchvision import transforms


def _load_stats(stats_path: str | Path | None) -> dict[str, Any]:
    if not stats_path:
        return {}
    path = Path(stats_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _take_or_pad(array: np.ndarray, length: int) -> np.ndarray:
    if array.shape[0] >= length:
        return array[:length]
    pad = np.repeat(array[-1:], length - array.shape[0], axis=0)
    return np.concatenate([array, pad], axis=0)


def _decode_sample(
    raw: dict[str, Any],
    transform: transforms.Compose,
    stats: dict[str, Any],
    T_obs: int,
    T_action: int,
) -> dict[str, Any]:
    images_np = np.load(io.BytesIO(raw["images.npy"])).astype(np.uint8)
    states_np = np.load(io.BytesIO(raw["states.npy"])).astype(np.float32)
    actions_np = np.load(io.BytesIO(raw["actions.npy"])).astype(np.float32)
    metadata = json.loads(raw.get("metadata.json", b"{}").decode("utf-8"))
    images_np = _take_or_pad(images_np, T_obs)
    states_np = _take_or_pad(states_np, T_obs)
    actions_np = _take_or_pad(actions_np, T_action)

    images = torch.stack([transform(Image.fromarray(image)) for image in images_np])
    states = torch.from_numpy(states_np)
    actions = torch.from_numpy(actions_np)
    action_stats = stats.get("actions", {})
    if "mean" in action_stats and "std" in action_stats:
        mean = torch.tensor(action_stats["mean"], dtype=actions.dtype)
        std = torch.tensor(action_stats["std"], dtype=actions.dtype).clamp_min(1e-6)
        actions = (actions - mean) / std

    return {
        "images": images,
        "states": states,
        "actions": actions,
        "mask": torch.ones(actions.shape[0], dtype=torch.bool),
        "language": metadata.get("language_instruction", ""),
        "episode": metadata.get("episode", raw.get("__key__", "")),
    }


def _make_transform(image_size: int, augment: bool) -> transforms.Compose:
    ops: list[Any] = []
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


def build_streaming_dataset(
    urls: str | list[str],
    image_size: int,
    T_obs: int,
    T_action: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    stats_path: str | Path | None,
    num_workers: int = 4,
    prefetch_factor: int = 4,
) -> wds.WebLoader:
    stats = _load_stats(stats_path)
    transform = _make_transform(image_size, augment)
    dataset = wds.WebDataset(urls, shardshuffle=100 if shuffle else False)
    if shuffle:
        dataset = dataset.shuffle(512)
    dataset = dataset.map(lambda sample: _decode_sample(sample, transform, stats, T_obs, T_action))
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return wds.WebLoader(dataset, **kwargs)


def seed_streaming(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
