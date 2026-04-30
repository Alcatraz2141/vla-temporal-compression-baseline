from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import webdataset as wds
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.vla_dataset import VLADataset


def _sample_to_np(sample: dict[str, object]) -> dict[str, object]:
    images = sample["images"].numpy()
    images = np.clip(np.transpose(images, (0, 2, 3, 1)) * 255.0, 0, 255).astype(np.uint8)
    return {
        "images.npy": images,
        "states.npy": sample["states"].numpy(),
        "actions.npy": sample["actions"].numpy(),
        "metadata.json": json.dumps(
            {
                "language_instruction": sample.get("language", ""),
                "episode": sample.get("episode", ""),
            }
        ).encode("utf-8"),
    }


def export_split(
    input_root: Path,
    output_root: Path,
    split: str,
    T_obs: int,
    T_action: int,
    image_size: int,
    stats_path: Path,
    max_samples_per_shard: int,
) -> None:
    dataset = VLADataset(
        root=input_root,
        split=split,
        T_obs=T_obs,
        T_action=T_action,
        image_size=image_size,
        augment=False,
        stats_path=None,
    )
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(split_dir / "shard-%06d.tar")
    with wds.ShardWriter(pattern, maxcount=max_samples_per_shard) as sink:
        for idx in tqdm(range(len(dataset)), desc=f"export {split}"):
            sample = _sample_to_np(dataset[idx])
            sample["__key__"] = f"{split}-{idx:09d}"
            sink.write(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export processed local VLA episodes to WebDataset shards.")
    parser.add_argument("--input-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("data/webdataset"))
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    parser.add_argument("--T-obs", type=int, default=4)
    parser.add_argument("--T-action", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--stats-path", type=Path, default=Path("data/processed/stats.json"))
    parser.add_argument("--max-samples-per-shard", type=int, default=2048)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stats_path.exists():
        target_stats = args.output_root / "stats.json"
        target_stats.write_bytes(args.stats_path.read_bytes())
    for split in args.splits:
        export_split(
            args.input_root,
            args.output_root,
            split,
            args.T_obs,
            args.T_action,
            args.image_size,
            args.stats_path,
            args.max_samples_per_shard,
        )


if __name__ == "__main__":
    main()
