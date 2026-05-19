from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.episode_loader import EpisodeDataset, episode_collate_fn
from datasets.vla_dataset import seed_worker


def source_root(source: str) -> Path:
    if source == "libero_long":
        return Path("data/libero_long")
    if source == "fractal":
        return Path("data/raw")
    return Path("data/raw_diverse")


def tensor_has_nan(batch: dict[str, Any]) -> bool:
    for value in batch.values():
        if torch.is_tensor(value) and torch.isnan(value.float()).any():
            return True
    return False


def batch_shapes(batch: dict[str, Any]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)}


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_source(source: str, args: argparse.Namespace) -> dict[str, object]:
    dataset = EpisodeDataset(
        source=source,
        split="train",
        K_recent=args.K_recent,
        H_action=args.H_action,
        max_memory_tokens=args.max_memory_tokens,
        chunk_size=args.chunk_size,
        seed=args.seed,
        root=args.data_root or source_root(source),
        split_dir=args.split_dir,
        image_size=args.image_size,
        samples_per_epoch=5,
        max_episodes=5,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(5, len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=episode_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    batch = next(iter(loader))
    shapes = batch_shapes(batch)
    print(f"\nsource={source}")
    print(json.dumps(shapes, indent=2))
    if tensor_has_nan(batch):
        raise ValueError(f"NaN detected in source={source}")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "batch_shape": json.dumps(shapes),
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test unified EpisodeDataset across sources.")
    parser.add_argument("--sources", nargs="*", default=["libero_long", "fractal", "ur5", "others"])
    parser.add_argument("--data-root", type=Path, default=None, help="Override root for all sources.")
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--K-recent", type=int, default=8)
    parser.add_argument("--H-action", type=int, default=4)
    parser.add_argument("--max-memory-tokens", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-path", type=Path, default=Path("results/smoke_test.csv"))
    args = parser.parse_args()

    failed = False
    for source in args.sources:
        try:
            row = run_source(source, args)
        except Exception as exc:
            traceback.print_exc()
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "batch_shape": "",
                "passed": False,
            }
            failed = True
        append_row(args.results_path, row)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
