from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


OFFICIAL_REPO = "https://github.com/google-deepmind/open_x_embodiment"
DEFAULT_DATASETS = [
    "fractal20220817_data",
    "bridge",
    "taco_play",
    "berkeley_autolab_ur5",
]


def make_synthetic_subset(root: Path, episodes: int, steps: int, image_size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for episode_idx in tqdm(range(episodes), desc="synthetic episodes"):
        episode_dir = root / f"episode_{episode_idx:06d}"
        image_dir = episode_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        actions = rng.normal(0, 0.25, size=(steps, 7)).astype(np.float32)
        states = rng.normal(0, 1.0, size=(steps, 8)).astype(np.float32)
        for t in range(steps):
            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            center = (int(image_size * (0.25 + 0.5 * t / max(steps - 1, 1))), int(image_size * 0.5))
            color = (60 + episode_idx * 17 % 180, 120, 220)
            cv2.circle(image, center, 16, color, -1)
            cv2.line(image, (0, image_size - 30), (image_size, image_size - 30), (80, 80, 80), 2)
            cv2.imwrite(str(image_dir / f"{t:06d}.jpg"), image)
        np.save(episode_dir / "actions.npy", actions)
        np.save(episode_dir / "states.npy", states)
        metadata = {
            "robot": "franka_panda",
            "embodiment": "Franka/Panda synthetic smoke test",
            "language_instruction": "move the end effector to the target",
        }
        with (episode_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def run_official_download(root: Path, datasets: list[str], max_episodes: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    print(f"Official Open X-Embodiment repo: {OFFICIAL_REPO}")
    print("Requested Franka/Panda-oriented datasets:")
    for name in datasets:
        print(f"  - {name}")
    print(
        "This script intentionally does not auto-download multi-terabyte RT-X shards. "
        "Clone the official repo and use its TFDS/RLDS builders or GCS instructions, "
        f"then cap preprocessing with --max-episodes {max_episodes}."
    )
    print("Example next step: datasets/preprocess.py --input-root data/raw --output-root data/processed")
    subprocess.run(["git", "ls-remote", "--heads", OFFICIAL_REPO], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire an Open X-Embodiment subset or create local smoke-test episodes.")
    parser.add_argument("--output-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--max-episodes", type=int, default=10_000)
    parser.add_argument("--synthetic", action="store_true", help="Create a tiny Franka/Panda-shaped local subset for smoke tests.")
    parser.add_argument("--synthetic-episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.synthetic:
        make_synthetic_subset(args.output_root, args.synthetic_episodes, args.steps, args.image_size, args.seed)
    else:
        run_official_download(args.output_root, args.datasets, args.max_episodes)


if __name__ == "__main__":
    main()
