from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MIX = {
    "berkeley_autolab_ur5": 1000,
    "roboturk": 1500,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": 1250,
    "bridge": 1250,
}


def run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_mix(items: list[str] | None) -> dict[str, int]:
    if not items:
        return dict(DEFAULT_MIX)
    out: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected DATASET=COUNT, got {item!r}")
        name, count = item.split("=", maxsplit=1)
        out[name] = int(count)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a non-Franka/diverse Open X subset without touching existing Franka data.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_diverse"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed_diverse"))
    parser.add_argument("--shards-root", type=Path, default=Path("data/episode_shards_diverse_5k"))
    parser.add_argument("--mix", nargs="*", default=None, help="Dataset mix as DATASET=COUNT. Defaults to UR5/Sawyer/Kuka/WidowX.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--episodes-per-shard", type=int, default=64)
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--download", action="store_true", help="Pass through to openx_rlds_to_raw.py. Usually avoid this; streaming is preferred.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mix = parse_mix(args.mix)
    if args.overwrite:
        import shutil

        for root in (args.raw_root, args.processed_root, args.shards_root):
            if root.exists() and not args.dry_run:
                shutil.rmtree(root)

    for dataset, count in mix.items():
        cmd = [
            sys.executable,
            "scripts/openx_rlds_to_raw.py",
            "--dataset",
            dataset,
            "--split",
            args.split,
            "--max-episodes",
            str(count),
            "--output-root",
            str(args.raw_root),
        ]
        if args.data_dir:
            cmd.extend(["--data-dir", args.data_dir])
        if args.download:
            cmd.append("--download")
        run(cmd, args.dry_run)

    run(
        [
            sys.executable,
            "datasets/preprocess.py",
            "--input-root",
            str(args.raw_root),
            "--output-root",
            str(args.processed_root),
            "--max-episodes",
            str(sum(mix.values())),
            "--image-size",
            str(args.image_size),
            "--stats-path",
            str(args.processed_root / "stats.json"),
            "--filter-keywords",
        ],
        args.dry_run,
    )
    run(
        [
            sys.executable,
            "scripts/export_episode_shards.py",
            "--input-root",
            str(args.processed_root),
            "--output-root",
            str(args.shards_root),
            "--max-episodes",
            str(sum(mix.values())),
            "--episodes-per-shard",
            str(args.episodes_per_shard),
            "--overwrite",
        ],
        args.dry_run,
    )


if __name__ == "__main__":
    main()
