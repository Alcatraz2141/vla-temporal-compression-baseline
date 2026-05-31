from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing h5py. Install with `uv add h5py`.") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def demo_keys(h5: Any) -> list[str]:
    if "data" in h5:
        return [f"data/{key}" for key in sorted(h5["data"].keys()) if isinstance(h5["data"][key], h5py.Group)]
    return [key for key in sorted(h5.keys()) if isinstance(h5[key], h5py.Group) and "actions" in h5[key]]


def parse_split_entry(entry: str) -> tuple[str, str] | None:
    parts = entry.split("::")
    if len(parts) < 3:
        return None
    return parts[-2], parts[-1]


def load_allowed(split_file: Path | None) -> set[tuple[str, str]] | None:
    if split_file is None:
        return None
    allowed: set[tuple[str, str]] = set()
    for line in split_file.read_text(encoding="utf-8").splitlines():
        parsed = parse_split_entry(line.strip())
        if parsed is not None:
            allowed.add(parsed)
    return allowed


def hdf5_files(root: Path, pattern: str) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.glob(pattern) if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LIBERO action statistics from HDF5 demos.")
    parser.add_argument("--data-root", type=Path, default=Path("data/libero_long"))
    parser.add_argument("--hdf5-glob", default="**/*.hdf5")
    parser.add_argument("--split-file", type=Path, default=Path("splits/libero_long_train.txt"))
    parser.add_argument("--output", type=Path, default=Path("results/libero_action_stats_train.json"))
    parser.add_argument(
        "--normalize-dims",
        default="0,1,2,3,4,5",
        help="Comma-separated action dims to normalize. Default excludes gripper dim 6.",
    )
    args = parser.parse_args()

    allowed = load_allowed(args.split_file) if args.split_file else None
    normalize_dims = [int(part.strip()) for part in args.normalize_dims.split(",") if part.strip()]
    count = 0
    demo_count = 0
    total: np.ndarray | None = None
    total_sq: np.ndarray | None = None
    min_action: np.ndarray | None = None
    max_action: np.ndarray | None = None
    gripper_positive = 0
    gripper_negative = 0
    gripper_zero = 0

    for file_path in hdf5_files(args.data_root, args.hdf5_glob):
        rel_path = file_path.relative_to(args.data_root).as_posix() if file_path.is_relative_to(args.data_root) else file_path.name
        with h5py.File(file_path, "r") as h5:
            for demo_key in demo_keys(h5):
                if allowed is not None and (rel_path, demo_key) not in allowed:
                    continue
                actions = np.asarray(h5[demo_key]["actions"], dtype=np.float64)
                if actions.ndim != 2 or actions.shape[0] == 0:
                    continue
                demo_count += 1
                count += actions.shape[0]
                batch_sum = actions.sum(axis=0)
                batch_sq = np.square(actions).sum(axis=0)
                total = batch_sum if total is None else total + batch_sum
                total_sq = batch_sq if total_sq is None else total_sq + batch_sq
                batch_min = actions.min(axis=0)
                batch_max = actions.max(axis=0)
                min_action = batch_min if min_action is None else np.minimum(min_action, batch_min)
                max_action = batch_max if max_action is None else np.maximum(max_action, batch_max)
                if actions.shape[1] >= 7:
                    gripper = actions[:, -1]
                    gripper_positive += int((gripper > 0).sum())
                    gripper_negative += int((gripper < 0).sum())
                    gripper_zero += int((gripper == 0).sum())

    if count == 0 or total is None or total_sq is None or min_action is None or max_action is None:
        raise SystemExit("No actions found. Check --data-root, --hdf5-glob, and --split-file.")

    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    result = {
        "data_root": str(args.data_root),
        "hdf5_glob": args.hdf5_glob,
        "split_file": str(args.split_file) if args.split_file else None,
        "demo_count": demo_count,
        "action_count": count,
        "actions": {
            "mean": mean.tolist(),
            "std": np.clip(std, 1e-6, None).tolist(),
            "min": min_action.tolist(),
            "max": max_action.tolist(),
            "normalize_dims": normalize_dims,
            "gripper_positive": gripper_positive,
            "gripper_negative": gripper_negative,
            "gripper_zero": gripper_zero,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
