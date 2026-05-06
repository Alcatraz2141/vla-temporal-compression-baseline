from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require_lerobot() -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "LeRobot is not installed. Run: uv sync --extra smolvla\n"
            "or: uv pip install 'lerobot[smolvla]'"
        ) from exc
    return LeRobotDataset


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _episode_dirs(input_root: Path, split: str) -> list[Path]:
    split_root = input_root / split
    root = split_root if split_root.exists() else input_root
    return sorted(p for p in root.glob("episode_*") if p.is_dir())


def export_split(
    input_root: Path,
    output_root: Path,
    split: str,
    repo_id: str,
    fps: int,
    image_key: str,
    state_key: str,
    action_key: str,
    robot_type: str,
    use_videos: bool,
    overwrite: bool,
    push: bool,
    private: bool,
) -> None:
    LeRobotDataset = _require_lerobot()
    episodes = _episode_dirs(input_root, split)
    if not episodes:
        raise FileNotFoundError(f"No episodes found under {input_root}/{split}")

    first_actions = np.load(episodes[0] / "actions.npy", mmap_mode="r")
    first_states = np.load(episodes[0] / "states.npy", mmap_mode="r")
    first_image = _read_rgb(sorted((episodes[0] / "images").glob("*"))[0])
    image_shape = tuple(first_image.shape)
    state_dim = int(first_states.shape[-1])
    action_dim = int(first_actions.shape[-1])

    local_root = output_root / split
    if overwrite and local_root.exists():
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        root=local_root,
        use_videos=use_videos,
        features={
            state_key: {"dtype": "float32", "shape": (state_dim,), "names": [f"s{i}" for i in range(state_dim)]},
            image_key: {"dtype": "video" if use_videos else "image", "shape": image_shape, "names": ["height", "width", "channel"]},
            action_key: {"dtype": "float32", "shape": (action_dim,), "names": [f"a{i}" for i in range(action_dim)]},
        },
    )

    for episode_dir in tqdm(episodes, desc=f"export lerobot {split}"):
        actions = np.load(episode_dir / "actions.npy").astype(np.float32)
        states = np.load(episode_dir / "states.npy").astype(np.float32)
        images = sorted((episode_dir / "images").glob("*"))
        metadata = _metadata(episode_dir)
        task = metadata.get("language_instruction", "open x manipulation task")
        length = min(len(images), len(actions), len(states))
        for idx in range(length):
            dataset.add_frame(
                {
                    state_key: states[idx],
                    image_key: _read_rgb(images[idx]),
                    action_key: actions[idx],
                    "task": task,
                }
            )
        dataset.save_episode()

    dataset.finalize()
    if push:
        dataset.push_to_hub(private=private, upload_large_folder=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export processed VLA episodes to LeRobotDataset format for SmolVLA.")
    parser.add_argument("--config", type=Path, default=Path("configs/smolvla.yaml"))
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--train-repo-id", type=str, default=None)
    parser.add_argument("--val-repo-id", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)["lerobot_export"]
    input_root = Path(cfg["input_root"])
    output_root = Path(cfg["output_root"])
    train_repo = args.train_repo_id or cfg.get("train_repo_id")
    val_repo = args.val_repo_id or cfg.get("val_repo_id")
    if args.split in {"train", "both"}:
        if not train_repo:
            raise ValueError("Set lerobot_export.train_repo_id or pass --train-repo-id.")
        export_split(
            input_root,
            output_root,
            cfg.get("train_split", "train"),
            train_repo,
            int(cfg.get("fps", 10)),
            cfg.get("image_key", "observation.images.front"),
            cfg.get("state_key", "observation.state"),
            cfg.get("action_key", "action"),
            cfg.get("robot_type", "franka_panda_openx"),
            bool(cfg.get("use_videos", True)),
            args.overwrite,
            args.push,
            bool(cfg.get("private", True)),
        )
    if args.split in {"val", "both"}:
        if not val_repo:
            raise ValueError("Set lerobot_export.val_repo_id or pass --val-repo-id.")
        export_split(
            input_root,
            output_root,
            cfg.get("val_split", "val"),
            val_repo,
            int(cfg.get("fps", 10)),
            cfg.get("image_key", "observation.images.front"),
            cfg.get("state_key", "observation.state"),
            cfg.get("action_key", "action"),
            cfg.get("robot_type", "franka_panda_openx"),
            bool(cfg.get("use_videos", True)),
            args.overwrite,
            args.push,
            bool(cfg.get("private", True)),
        )


if __name__ == "__main__":
    main()
