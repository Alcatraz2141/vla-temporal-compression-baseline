#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# run_all_rollouts.sh — Run LIBERO rollouts for all 4 models x 3 seeds
#
# Usage:
#   bash libero_rollout_env/run_all_rollouts.sh
#
# Expects checkpoints at:
#   checkpoints/libero_long/                  (seed 42)
#   checkpoints/libero_long_seed43/           (seed 43)
#   checkpoints/libero_long_seed44/           (seed 44)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="${REPO_ROOT}/libero_rollout_env/run_rollout.sh"

EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
MAX_STEPS="${MAX_STEPS:-300}"
TASKS="${TASKS:-all}"

declare -A CONFIGS
CONFIGS["sliding_window"]="configs/libero_long_sliding_window.yaml"
CONFIGS["event_gated_memory"]="configs/libero_long_event_gated.yaml"
CONFIGS["age_gated_memory"]="configs/ablation_gate_age.yaml"
CONFIGS["event_gated_concat_query"]="configs/ablation_query_concat.yaml"

SEEDS=(42 43 44)
SEED_DIRS=("libero_long" "libero_long_seed43" "libero_long_seed44")

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    seed_dir="${SEED_DIRS[$i]}"
    for run_name in sliding_window event_gated_memory age_gated_memory event_gated_concat_query; do
        config="${CONFIGS[$run_name]}"
        ckpt="checkpoints/${seed_dir}/${run_name}/best.pt"
        if [ ! -f "${REPO_ROOT}/${ckpt}" ]; then
            echo "SKIP: ${ckpt} not found"
            continue
        fi
        echo ""
        echo "====== ${run_name} | seed=${seed} ======"
        bash "${RUNNER}" "${config}" "${ckpt}" \
            --tasks "${TASKS}" \
            --episodes-per-task "${EPISODES_PER_TASK}" \
            --max-steps "${MAX_STEPS}" \
            --seed "${seed}"
    done
done

echo ""
echo "All rollouts complete. Results at: results/libero_rollouts.csv"
