from __future__ import annotations

import argparse
import json
import random
import tarfile
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


def write_split(path: Path, episode_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"episodes": episode_ids}, f, indent=2)


def add_episode_to_tar(tar: tarfile.TarFile, src: Path, episode_id: str, metadata: dict[str, Any]) -> None:
    tar.add(src / "actions.npy", arcname=f"{episode_id}/actions.npy")
    tar.add(src / "states.npy", arcname=f"{episode_id}/states.npy")
    metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8")
    info = tarfile.TarInfo(name=f"{episode_id}/metadata.json")
    info.size = len(metadata_bytes)
    import io

    tar.addfile(info, io.BytesIO(metadata_bytes))
    for image_path in sorted((src / "images").glob("*")):
        if image_path.is_file():
            tar.add(image_path, arcname=f"{episode_id}/images/{image_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export processed episodes as episode-level tar shards.")
    parser.add_argument("--input-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("data/episode_shards"))
    parser.add_argument("--episodes-per-shard", type=int, default=64)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite and args.output_root.exists():
        import shutil

        shutil.rmtree(args.output_root)
    shards_out = args.output_root / "shards"
    splits_out = args.output_root / "splits"
    shards_out.mkdir(parents=True, exist_ok=True)
    splits_out.mkdir(parents=True, exist_ok=True)

    src_episodes = episode_dirs(args.input_root)
    if args.max_episodes is not None:
        src_episodes = src_episodes[: args.max_episodes]
    if not src_episodes:
        raise FileNotFoundError(f"No episode_* folders found under {args.input_root}")

    manifest: list[dict[str, Any]] = []
    skipped = 0
    shard: tarfile.TarFile | None = None
    shard_idx = -1
    in_shard = 0
    for src in tqdm(src_episodes, desc="export episode shards"):
        length = episode_length(src)
        if length < 2:
            skipped += 1
            continue
        if shard is None or in_shard >= args.episodes_per_shard:
            if shard is not None:
                shard.close()
            shard_idx += 1
            in_shard = 0
            shard = tarfile.open(shards_out / f"shard-{shard_idx:06d}.tar", "w")
        episode_id = f"episode_{len(manifest):06d}"
        metadata = read_json(src / "metadata.json")
        metadata.setdefault("source_episode", src.name)
        metadata.setdefault("episode_id", episode_id)
        metadata["length"] = length
        add_episode_to_tar(shard, src, episode_id, metadata)
        manifest.append(
            {
                "episode_id": episode_id,
                "shard": f"shards/shard-{shard_idx:06d}.tar",
                "length": length,
                "metadata": metadata,
            }
        )
        in_shard += 1
    if shard is not None:
        shard.close()
    if not manifest:
        raise ValueError("No valid episodes were exported.")

    rng = random.Random(args.seed)
    episode_ids = [item["episode_id"] for item in manifest]
    rng.shuffle(episode_ids)
    n_total = len(episode_ids)
    n_train = int(n_total * args.train_frac)
    n_val = int(n_total * args.val_frac)
    if n_total >= 3:
        n_val = max(n_val, 1)
        n_train = min(max(n_train, 1), n_total - n_val - 1)
    write_split(splits_out / "train.json", episode_ids[:n_train])
    write_split(splits_out / "val.json", episode_ids[n_train : n_train + n_val])
    write_split(splits_out / "test.json", episode_ids[n_train + n_val :])

    with (args.output_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")
    print(
        f"Wrote {len(manifest)} episodes into {shard_idx + 1} shard(s) under {args.output_root}; "
        f"skipped {skipped} too-short/invalid episodes"
    )


if __name__ == "__main__":
    main()
