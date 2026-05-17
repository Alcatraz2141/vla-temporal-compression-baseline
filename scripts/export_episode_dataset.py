from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    for split in ("train", "val", "test"):
        split_root = root / split
        if split_root.exists():
            dirs.extend(sorted(p for p in split_root.glob("episode_*") if p.is_dir()))
    if not dirs:
        dirs = sorted(p for p in root.glob("episode_*") if p.is_dir())
    return dirs


def episode_length(src: Path) -> int:
    try:
        import numpy as np

        actions_len = len(np.load(src / "actions.npy", mmap_mode="r"))
        states_len = len(np.load(src / "states.npy", mmap_mode="r"))
    except Exception:
        return 0
    images_len = len(list((src / "images").glob("*")))
    return min(actions_len, states_len, images_len)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def write_split(path: Path, episode_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"episodes": episode_ids}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export local processed episodes into a compact episode-level dataset layout.")
    parser.add_argument("--input-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("data/episodes"))
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking to save source independence at the cost of storage.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    episodes_out = args.output_root / "episodes"
    splits_out = args.output_root / "splits"
    episodes_out.mkdir(parents=True, exist_ok=True)
    splits_out.mkdir(parents=True, exist_ok=True)

    src_episodes = episode_dirs(args.input_root)
    if args.max_episodes is not None:
        src_episodes = src_episodes[: args.max_episodes]
    if not src_episodes:
        raise FileNotFoundError(f"No episode_* folders found under {args.input_root}")

    manifest: list[dict[str, Any]] = []
    skipped = 0
    for src in tqdm(src_episodes, desc="export episodes"):
        length = episode_length(src)
        if length < 2:
            skipped += 1
            continue
        episode_id = f"episode_{len(manifest):06d}"
        dst = episodes_out / episode_id
        dst.mkdir(parents=True, exist_ok=True)
        link_or_copy(src / "images", dst / "images", args.copy)
        link_or_copy(src / "actions.npy", dst / "actions.npy", args.copy)
        link_or_copy(src / "states.npy", dst / "states.npy", args.copy)
        metadata = read_json(src / "metadata.json")
        metadata.setdefault("source_episode", src.name)
        metadata.setdefault("episode_id", episode_id)
        with (dst / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        manifest.append({"episode_id": episode_id, "path": str(dst), "length": length, "metadata": metadata})

    rng = random.Random(args.seed)
    episode_ids = [item["episode_id"] for item in manifest]
    rng.shuffle(episode_ids)
    n_total = len(episode_ids)
    n_train = int(n_total * args.train_frac)
    n_val = int(n_total * args.val_frac)
    if n_total >= 3:
        n_val = max(n_val, 1)
        n_train = min(max(n_train, 1), n_total - n_val - 1)
    train_ids = episode_ids[:n_train]
    val_ids = episode_ids[n_train : n_train + n_val]
    test_ids = episode_ids[n_train + n_val :]
    write_split(splits_out / "train.json", train_ids)
    write_split(splits_out / "val.json", val_ids)
    write_split(splits_out / "test.json", test_ids)
    with (args.output_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")
    print(
        f"Wrote {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test episodes under {args.output_root}; "
        f"skipped {skipped} too-short/invalid episodes"
    )


if __name__ == "__main__":
    main()
