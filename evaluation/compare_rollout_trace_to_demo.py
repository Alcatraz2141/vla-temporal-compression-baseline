from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.rollout_alignment_checks import demo_arrays, task_file

LIBERO_10_TASKS = [
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
]


def _load_trace(path: Path, task_id: int, episode_idx: int) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["task_id"]) == task_id and int(row["episode_idx"]) == episode_idx:
                rows.append(row)
    if not rows:
        raise ValueError(f"No trace rows found for task_id={task_id}, episode_idx={episode_idx} in {path}")

    actions = np.asarray([[float(row[f"action_{i}"]) for i in range(7)] for row in rows], dtype=np.float32)
    pre_states = np.asarray([[float(row[f"pre_state_{i}"]) for i in range(8)] for row in rows], dtype=np.float32)
    post_states = np.asarray([[float(row[f"post_state_{i}"]) for i in range(8)] for row in rows], dtype=np.float32)
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float32)
    return {"actions": actions, "pre_states": pre_states, "post_states": post_states, "rewards": rewards}


def _first_close(actions: np.ndarray) -> int | None:
    close = np.flatnonzero(actions[:, -1] < 0.0)
    return int(close[0]) if close.size else None


def _first_positive(actions: np.ndarray) -> int | None:
    positive = np.flatnonzero(actions[:, -1] >= 0.0)
    return int(positive[0]) if positive.size else None


def _transitions(actions: np.ndarray) -> list[int]:
    sign = actions[:, -1] >= 0.0
    return [int(i) for i in np.flatnonzero(sign[1:] != sign[:-1]) + 1]


def _window(values: np.ndarray, center: int | None, radius: int) -> list[list[float]]:
    if center is None:
        return []
    start = max(0, center - radius)
    end = min(len(values), center + radius + 1)
    return values[start:end].round(5).tolist()


def compare(trace_path: Path, data_root: Path, task_id: int, episode_idx: int, window_radius: int) -> dict[str, Any]:
    trace = _load_trace(trace_path, task_id, episode_idx)
    task_name = LIBERO_10_TASKS[task_id]
    h5_path = task_file(data_root, task_name)
    _images, expert_states, expert_actions = demo_arrays(h5_path, episode_idx)

    rollout_actions = trace["actions"]
    rollout_states = trace["pre_states"]
    rollout_first_close = _first_close(rollout_actions)
    expert_first_close = _first_close(expert_actions)
    rollout_first_positive = _first_positive(rollout_actions)
    expert_first_positive = _first_positive(expert_actions)

    result: dict[str, Any] = {
        "trace_path": str(trace_path),
        "h5_path": str(h5_path),
        "task_id": task_id,
        "task_name": task_name,
        "episode_idx": episode_idx,
        "rollout_steps": int(len(rollout_actions)),
        "expert_steps": int(len(expert_actions)),
        "rollout_success": bool(np.any(trace["rewards"] > 0.0)),
        "rollout_first_close_step": rollout_first_close,
        "expert_first_close_step": expert_first_close,
        "rollout_first_positive_gripper_step": rollout_first_positive,
        "expert_first_positive_gripper_step": expert_first_positive,
        "rollout_gripper_close_fraction": float(np.mean(rollout_actions[:, -1] < 0.0)),
        "expert_gripper_close_fraction": float(np.mean(expert_actions[:, -1] < 0.0)),
        "rollout_gripper_transitions": _transitions(rollout_actions),
        "expert_gripper_transitions": _transitions(expert_actions),
        "rollout_action_window_around_first_close": _window(rollout_actions, rollout_first_close, window_radius),
        "expert_action_window_around_first_close": _window(expert_actions, expert_first_close, window_radius),
        "rollout_action_window_around_first_positive": _window(rollout_actions, rollout_first_positive, window_radius),
        "expert_action_window_around_first_positive": _window(expert_actions, expert_first_positive, window_radius),
    }

    if rollout_first_close is not None and expert_first_close is not None:
        rollout_close_pos = rollout_states[rollout_first_close, :3]
        expert_close_pos = expert_states[expert_first_close, :3]
        result["rollout_eef_pos_at_first_close"] = rollout_close_pos.round(6).tolist()
        result["expert_eef_pos_at_first_close"] = expert_close_pos.round(6).tolist()
        result["eef_pos_error_at_first_close"] = float(np.linalg.norm(rollout_close_pos - expert_close_pos))

        dists = np.linalg.norm(rollout_states[:, :3] - expert_close_pos.reshape(1, 3), axis=1)
        closest_step = int(np.argmin(dists))
        result["closest_rollout_step_to_expert_close_pos"] = closest_step
        result["closest_rollout_dist_to_expert_close_pos"] = float(dists[closest_step])
        result["closest_rollout_eef_pos"] = rollout_states[closest_step, :3].round(6).tolist()
        result["rollout_close_minus_expert_close_steps"] = int(rollout_first_close - expert_first_close)

    limit = min(len(rollout_actions), len(expert_actions))
    if limit:
        result["prefix_position_mse"] = float(np.mean((rollout_actions[:limit, :3] - expert_actions[:limit, :3]) ** 2))
        result["prefix_rotation_mse"] = float(np.mean((rollout_actions[:limit, 3:6] - expert_actions[:limit, 3:6]) ** 2))
        result["prefix_gripper_sign_accuracy"] = float(np.mean((rollout_actions[:limit, -1] >= 0.0) == (expert_actions[:limit, -1] >= 0.0)))

    if rollout_first_positive is not None and expert_first_positive is not None:
        rollout_pos = rollout_states[rollout_first_positive, :3]
        expert_pos = expert_states[expert_first_positive, :3]
        result["rollout_eef_pos_at_first_positive_gripper"] = rollout_pos.round(6).tolist()
        result["expert_eef_pos_at_first_positive_gripper"] = expert_pos.round(6).tolist()
        result["eef_pos_error_at_first_positive_gripper"] = float(np.linalg.norm(rollout_pos - expert_pos))
        result["rollout_positive_minus_expert_positive_steps"] = int(rollout_first_positive - expert_first_positive)

        dists = np.linalg.norm(rollout_states[:, :3] - expert_pos.reshape(1, 3), axis=1)
        closest_step = int(np.argmin(dists))
        result["closest_rollout_step_to_expert_positive_gripper_pos"] = closest_step
        result["closest_rollout_dist_to_expert_positive_gripper_pos"] = float(dists[closest_step])
        result["closest_rollout_eef_pos_to_expert_positive_gripper"] = rollout_states[closest_step, :3].round(6).tolist()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare an online rollout trace with the matching LIBERO HDF5 demo.")
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/libero_long"))
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode-idx", type=int, required=True)
    parser.add_argument("--window-radius", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = compare(args.trace_path, args.data_root, args.task_id, args.episode_idx, args.window_radius)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
