from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import h5py
except ImportError as exc:
    raise SystemExit("Missing h5py. Install with `uv add h5py`. For official LIBERO tools, pip install libero.") from exc

try:
    import libero  # noqa: F401
except ImportError:
    libero = None


STATE_KEY_GROUPS = (
    ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
    ("ee_pos", "ee_ori", "gripper_states"),
    ("ee_states", "gripper_states"),
    ("joint_states", "gripper_states"),
)


def state_keys(obs):
    for keys in STATE_KEY_GROUPS:
        if all(key in obs for key in keys):
            return keys
    return tuple(key for key in ("ee_pos", "ee_ori", "ee_states", "gripper_states", "joint_states") if key in obs)


def find_hdf5(root: Path, pattern: str) -> Path:
    if root.is_file():
        return root
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No HDF5 files found under {root} with pattern {pattern!r}. pip install libero and download LIBERO demos first.")
    return matches[0]


def demo_keys(h5: Any) -> list[str]:
    if "data" in h5:
        return [f"data/{key}" for key in sorted(h5["data"].keys()) if isinstance(h5["data"][key], h5py.Group)]
    return [key for key in sorted(h5.keys()) if isinstance(h5[key], h5py.Group) and "obs" in h5[key]]


def language(h5: Any, demo: Any, fallback: str) -> str:
    for source in (demo.attrs, h5.attrs):
        for key in ("language_instruction", "language", "task_description", "instruction"):
            if key in source:
                value = source[key]
                return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    if "env_args" in h5.attrs:
        try:
            env_args = json.loads(h5.attrs["env_args"])
            for key in ("language_instruction", "task_description", "problem_name", "task"):
                if key in env_args:
                    return str(env_args[key])
        except Exception:
            pass
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one LIBERO HDF5 episode and print keys/shapes.")
    parser.add_argument("--data-root", type=Path, default=Path("data/libero_long"))
    parser.add_argument("--hdf5-glob", default="**/*.hdf5")
    parser.add_argument("--episode-index", type=int, default=0)
    args = parser.parse_args()

    path = find_hdf5(args.data_root, args.hdf5_glob)
    print(f"LIBERO import: {'available' if libero is not None else 'not installed; direct h5py inspection only (pip install libero)'}")
    print(f"file: {path}")
    with h5py.File(path, "r") as h5:
        print(f"root keys: {list(h5.keys())}")
        keys = demo_keys(h5)
        if not keys:
            raise ValueError("No demo groups found. Expected data/demo_* with obs/actions.")
        demo_key = keys[min(args.episode_index, len(keys) - 1)]
        demo = h5[demo_key]
        obs = demo["obs"]
        print(f"episode key: {demo_key}")
        print(f"obs keys: {list(obs.keys())}")
        image = obs["agentview_rgb"]
        actions = demo["actions"]
        keys = state_keys(obs)
        state_shapes = {key: obs[key].shape for key in keys}
        state_dim = sum(obs[key].shape[-1] for key in keys)
        print(f"image shape: {image.shape}, dtype={image.dtype}")
        print(f"action shape: {actions.shape}, dtype={actions.dtype}")
        print(f"state/proprio component shapes: {state_shapes}")
        print(f"state/proprio dim: {state_dim}")
        print(f"action dim: {actions.shape[-1]}")
        print(f"selected state keys: {keys}")
        print(f"episode length: {min(len(image), len(actions), *(obs[key].shape[0] for key in keys))}")
        print(f"language instruction: {language(h5, demo, path.stem)}")


if __name__ == "__main__":
    main()
