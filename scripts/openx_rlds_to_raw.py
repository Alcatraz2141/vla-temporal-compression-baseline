#!/usr/bin/env python3
"""Stream Open X–style RLDS episodes from TensorFlow Datasets into this repo's local layout.

Writes ``data/raw/episode_XXXXXX/{images/*.jpg,actions.npy,states.npy,metadata.json}`` so you can
run ``datasets/preprocess.py`` and ``scripts/export_webdataset.py`` next.

Requires the optional Open X stack (TensorFlow + TFDS). Use **Python 3.11 or 3.12** (``openx`` pins
protobuf for TFDS; that combination only has TensorFlow wheels up through 3.12, not 3.13+). Example::

    uv sync --extra openx
    uv run python scripts/openx_rlds_to_raw.py \\
        --dataset fractal20220817_data \\
        --max-episodes 100 \\
        --output-root data/raw

By default, RLDS **importer** datasets (e.g. ``fractal20220817_data``) **stream from public GCS**
(``gs://gresearch/robotics/...``) without Beam—no ``--download`` and no local TFDS cache required.

Optional ``--download`` materializes the full split under ``~/tensorflow_datasets`` via Apache Beam
(heavy disk + time; needs ``apache-beam``).

Subset knob: ``--max-episodes`` caps how many episodes are materialized (streaming; does not
require downloading the full split first).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
except ImportError as e:
    print(
        "Missing dependency. Install with: uv sync --extra openx\n"
        "(requires a Python version supported by TensorFlow.)",
        file=sys.stderr,
    )
    raise SystemExit(1) from e


IMAGE_KEYS = (
    "image",
    "image_0",
    "rgb_static",
    "image_static",
    "hand_image",
    "image_wrist",
    "workspace",
)

STATE_KEYS = (
    "proprio",
    "state",
    "eef_state",
    "base_pose_tool_reached",
    "tcp_pose",
    "cartesian_position",
    "joint_positions",
)

ACTION_ORDER = (
    "world_vector",
    "rotation_delta",
    "gripper_closedness_action",
)


def pad_truncate(vec: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size >= dim:
        return v[:dim].astype(np.float32)
    out = np.zeros(dim, dtype=np.float32)
    out[: v.size] = v
    return out


def _decode_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "SU":
            return str(x.item()) if x.size == 1 else str(x)
        if x.dtype == object and x.size == 1:
            return _decode_text(x.item())
    return str(x)


def _pick_image(obs: dict[str, Any]) -> np.ndarray | None:
    if not isinstance(obs, dict):
        return None
    for key in IMAGE_KEYS:
        if key not in obs:
            continue
        img = obs[key]
        if img is None:
            continue
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
            continue
        if arr.dtype in (np.float32, np.float64):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr
    return None


def _flatten_state(obs: dict[str, Any]) -> np.ndarray:
    if not isinstance(obs, dict):
        return np.zeros(8, dtype=np.float32)
    for key in STATE_KEYS:
        if key not in obs:
            continue
        v = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        if v.size > 0:
            return pad_truncate(v, 8)
    return np.zeros(8, dtype=np.float32)


def _flatten_action(action: Any) -> np.ndarray:
    if action is None:
        return np.zeros(7, dtype=np.float32)
    if isinstance(action, dict):
        parts: list[np.ndarray] = []
        for key in ACTION_ORDER:
            if key not in action:
                continue
            parts.append(np.asarray(action[key], dtype=np.float32).reshape(-1))
        if parts:
            return pad_truncate(np.concatenate(parts), 7)
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    return pad_truncate(arr, 7)


def _language_from_step(step: dict[str, Any]) -> str:
    for path in (
        ("language_instruction",),
        ("observation", "language_instruction"),
        ("observation", "natural_language_instruction"),
        ("natural_language_instruction",),
    ):
        cur: Any = step
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if cur is not None:
            text = _decode_text(cur)
            if text.strip():
                return text.strip()
    return ""


def _write_episode(
    out_dir: Path,
    episode_index: int,
    steps: list[dict[str, Any]],
    dataset_name: str,
    assume_rgb: bool,
) -> bool:
    if not steps:
        return False
    ep_dir = out_dir / f"episode_{episode_index:06d}"
    img_dir = ep_dir / "images"
    if ep_dir.exists():
        shutil.rmtree(ep_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    actions_list: list[np.ndarray] = []
    states_list: list[np.ndarray] = []
    lang = ""
    frame_idx = 0

    for step in steps:
        obs = step.get("observation", step)
        if not isinstance(obs, dict):
            obs = {}
        img = _pick_image(obs)
        if img is None:
            continue
        if not lang:
            lang = _language_from_step(step)
        if assume_rgb:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        cv2.imwrite(str(img_dir / f"{frame_idx:06d}.jpg"), img_bgr)
        frame_idx += 1
        act = step.get("action")
        actions_list.append(_flatten_action(act))
        states_list.append(_flatten_state(obs))

    if not actions_list:
        shutil.rmtree(ep_dir, ignore_errors=True)
        return False

    actions = np.stack(actions_list, axis=0)
    states = np.stack(states_list, axis=0)
    np.save(ep_dir / "actions.npy", actions)
    np.save(ep_dir / "states.npy", states)
    meta = {
        "robot": "open_x_rlds",
        "embodiment": f"{dataset_name} (Open X RLDS)",
        "language_instruction": lang or "unknown task",
        "source_dataset": dataset_name,
        "source_format": "rlds",
    }
    with (ep_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return True


def _iter_episode_steps(episode: dict[str, Any]) -> list[dict[str, Any]]:
    steps = episode["steps"]
    if isinstance(steps, tf.data.Dataset):
        return list(steps.as_numpy_iterator())
    if isinstance(steps, dict):
        return [tf.nest.map_structure(lambda x: x, steps)]
    raise TypeError(f"Unexpected steps type: {type(steps)}")


def _open_episode_stream(
    builder: Any,
    split: str,
    max_episodes: int,
    *,
    download: bool,
) -> Any:
    """Return a tf.data.Dataset of RLDS episodes (already ``.take(max_episodes)``)."""
    if download:
        print(
            "Running download_and_prepare() (Apache Beam) — large download + disk use; can take a long time …",
            flush=True,
        )
        builder.download_and_prepare()
        return builder.as_dataset(split=split, shuffle_files=False).take(max_episodes)

    try:
        return builder.as_dataset(split=split, shuffle_files=False).take(max_episodes)
    except AssertionError as e:
        msg = str(e).lower()
        if (
            "could not find data" not in msg
            and "download_and_prepare" not in msg
            and "download=true" not in msg
        ):
            raise

    gcs_fn = getattr(builder, "get_dataset_location", None)
    gcs = gcs_fn() if callable(gcs_fn) else None
    if isinstance(gcs, str) and gcs.startswith("gs://"):
        print(
            f"No local TFDS cache — streaming up to {max_episodes} episode(s) from:\n  {gcs}",
            flush=True,
        )
        src = tfds.builder_from_directory(gcs)
        return src.as_dataset(split=split, shuffle_files=False).take(max_episodes)

    print(
        "No local TFDS data and this builder has no GCS stream path.\n"
        "  • Pass --download to materialize with Apache Beam (very large), or\n"
        "  • Mirror the dataset and pass --data-dir.\n",
        file=sys.stderr,
    )
    raise SystemExit(4) from e


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize RLDS episodes from TFDS into data/raw for this baseline.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="TFDS builder name (e.g. fractal20220817_data). See tensorflow.org/datasets/catalog.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="TFDS split to read (default: train).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=100,
        help="Maximum number of episodes to write (subset knob).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw"),
        help="Directory to create episode_* folders under.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="TFDS root directory (default: ~/tensorflow_datasets). Use after gsutil mirror.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Beam: materialize the full split under ~/tensorflow_datasets (huge); default is GCS streaming instead.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing episode_* under output-root before writing.",
    )
    parser.add_argument(
        "--assume-rgb",
        action="store_true",
        default=True,
        help="Treat loaded images as RGB and convert to BGR for cv2.imwrite (default: on).",
    )
    parser.add_argument(
        "--no-assume-rgb",
        action="store_false",
        dest="assume_rgb",
        help="Write image arrays as-is (already BGR).",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for p in sorted(args.output_root.glob("episode_*")):
            if p.is_dir():
                shutil.rmtree(p)

    print(f"Loading builder {args.dataset!r} (split={args.split!r}) …", flush=True)
    try:
        builder = tfds.builder(args.dataset, data_dir=args.data_dir)
    except Exception as e:
        err = type(e).__name__
        if "NotFound" not in err and "not found" not in str(e).lower():
            raise
        print(
            "Dataset not registered. Install TFDS configs from open_x_embodiment or mirror GCS:\n"
            "  gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/<dataset_name> ~/tensorflow_datasets/\n"
            "then pass --data-dir pointing at the parent of the dataset folder.",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    ds = _open_episode_stream(
        builder,
        args.split,
        args.max_episodes,
        download=args.download,
    )
    written = 0
    episode_index = 0
    for episode in ds:
        steps = _iter_episode_steps(episode)
        if _write_episode(args.output_root, episode_index, steps, args.dataset, args.assume_rgb):
            written += 1
            print(
                f"Wrote {args.output_root / f'episode_{episode_index:06d}'} ({len(steps)} RLDS steps)",
                flush=True,
            )
        episode_index += 1

    if written == 0:
        print("No episodes written. Check dataset schema / image keys.", file=sys.stderr)
        raise SystemExit(3)
    print(f"Done. {written} episode(s) under {args.output_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
