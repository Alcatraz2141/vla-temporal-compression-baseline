#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/workspace/run_backups}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="${OUT_DIR}/vla_run_artifacts_${STAMP}.tar.gz"

mkdir -p "${OUT_DIR}"

tar -czf "${OUT}" \
  checkpoints \
  logs \
  results \
  configs \
  splits \
  AGENTS.md \
  experimentation.md \
  libero_rollout_env \
  libero_config \
  utils/language.py

echo "${OUT}"
ls -lh "${OUT}"
