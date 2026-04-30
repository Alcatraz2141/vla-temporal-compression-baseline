from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


FRANKA_KEYWORDS = ("franka", "panda", "fractal", "bridge", "taco", "berkeley")


def _episode_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.glob("episode_*") if p.is_dir()])


def _read_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_franka_or_panda(metadata: dict[str, Any], episode_dir: Path) -> bool:
    text = " ".join(str(v).lower() for v in metadata.values()) + " " + episode_dir.name.lower()
    return any(keyword in text for keyword in FRANKA_KEYWORDS)


def _resize_images(src_dir: Path, dst_dir: Path, image_size: int) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(sorted(src_dir.glob("*"))):
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_dir / f"{idx:06d}.jpg"), image)


def _copy_local_episode(src: Path, dst: Path, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    dst.mkdir(parents=True, exist_ok=True)
    actions = np.load(src / "actions.npy").astype(np.float32)
    states = np.load(src / "states.npy").astype(np.float32)
    np.save(dst / "actions.npy", actions)
    np.save(dst / "states.npy", states)
    _resize_images(src / "images", dst / "images", image_size)
    metadata = _read_metadata(src)
    metadata.setdefault("source_episode", src.name)
    with (dst / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return actions, states


def _write_stats(actions_list: list[np.ndarray], states_list: list[np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    actions = np.concatenate(actions_list, axis=0)
    states = np.concatenate(states_list, axis=0)
    stats = {
        "actions": {"mean": actions.mean(axis=0).tolist(), "std": actions.std(axis=0).clip(min=1e-6).tolist()},
        "states": {"mean": states.mean(axis=0).tolist(), "std": states.std(axis=0).clip(min=1e-6).tolist()},
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def preprocess_local(
    input_root: Path,
    output_root: Path,
    max_episodes: int,
    image_size: int,
    val_fraction: float,
    stats_path: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    actions_list: list[np.ndarray] = []
    states_list: list[np.ndarray] = []
    candidates = []
    for episode_dir in _episode_dirs(input_root):
        metadata = _read_metadata(episode_dir)
        if _is_franka_or_panda(metadata, episode_dir):
            candidates.append(episode_dir)
    selected = candidates[:max_episodes]
    if not selected:
        raise FileNotFoundError(f"No Franka/Panda-like episodes found under {input_root}")

    val_count = max(1, int(len(selected) * val_fraction)) if len(selected) > 1 else 0
    for idx, src in enumerate(tqdm(selected, desc="preprocess")):
        split = "val" if idx < val_count else "train"
        dst = output_root / split / f"episode_{idx:06d}"
        actions, states = _copy_local_episode(src, dst, image_size)
        actions_list.append(actions)
        states_list.append(states)
    _write_stats(actions_list, states_list, stats_path)


def preprocess_tfrecords(_: Path, __: Path) -> None:
    raise NotImplementedError(
        "Native RLDS/TFRecord conversion requires TensorFlow Datasets and the Open X-Embodiment builders. "
        "Install those optional dependencies, materialize episodes with the official RT-X scripts, then run "
        "this preprocessor in local_episode mode."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Open X-Embodiment/RT-X episodes into local PyTorch-friendly layout.")
    parser.add_argument("--input-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--format", choices=["local_episode", "tfrecord"], default="local_episode")
    parser.add_argument("--max-episodes", type=int, default=10_000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--stats-path", type=Path, default=Path("data/processed/stats.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    if args.format == "tfrecord":
        preprocess_tfrecords(args.input_root, args.output_root)
    else:
        preprocess_local(args.input_root, args.output_root, args.max_episodes, args.image_size, args.val_fraction, args.stats_path)


if __name__ == "__main__":
    main()
