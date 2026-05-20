#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# run_rollout.sh — Run LIBERO sim rollouts using the isolated venv
#
# Usage:
#   bash libero_rollout_env/run_rollout.sh <config> <checkpoint> [extra args...]
#
# Examples:
#   # Single task smoke test
#   bash libero_rollout_env/run_rollout.sh \
#     configs/libero_long_sliding_window.yaml \
#     checkpoints/libero_long/sliding_window/best.pt \
#     --tasks 0 --episodes-per-task 1 --max-steps 50
#
#   # Full 10-task rollout (all init states)
#   bash libero_rollout_env/run_rollout.sh \
#     configs/ablation_query_concat.yaml \
#     checkpoints/libero_long/event_gated_concat_query/best.pt \
#     --tasks all --episodes-per-task 50 --max-steps 300
#
#   # Multi-seed sweep
#   for seed in 42 43 44; do
#     bash libero_rollout_env/run_rollout.sh \
#       configs/libero_long_event_gated.yaml \
#       checkpoints/libero_long_seed${seed}/event_gated_memory/best.pt \
#       --tasks all --episodes-per-task 50 --seed $seed
#   done
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${WORKSPACE:-/workspace}/libero_rollout_envs/.venv"
LIBERO_SRC="${WORKSPACE:-/workspace}/libero_rollout_envs/LIBERO"
PYTHON="${VENV}/bin/python"

if [ ! -f "${PYTHON}" ]; then
    echo "ERROR: Rollout venv not found at ${VENV}"
    echo "Run:  bash libero_rollout_env/bootstrap.sh"
    exit 1
fi

if [ $# -lt 2 ]; then
    echo "Usage: bash libero_rollout_env/run_rollout.sh <config> <checkpoint> [extra args...]"
    exit 1
fi

CONFIG="$1"
CHECKPOINT="$2"
shift 2

export LIBERO_CONFIG_PATH="${REPO_ROOT}/libero_config"
export PYTHONPATH="${LIBERO_SRC}:${REPO_ROOT}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

exec "${PYTHON}" "${REPO_ROOT}/evaluation/libero_rollout.py" \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    "$@"
