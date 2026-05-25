# LIBERO Online Rollout Environment

Isolated `uv` environment for running trained policies in the official LIBERO simulator.
Completely separate from the main project's `uv` environment — no dependency conflicts.

## RunPod Setup (one-time)

```bash
# 1. Clone repo and sync main env (for training/eval — unchanged)
cd /root
git clone <repo-url> vla-temporal-compression-baseline
cd vla-temporal-compression-baseline
uv sync

# 2. Bootstrap the LIBERO rollout env (installs into /workspace, persistent)
bash libero_rollout_env/bootstrap.sh
```

Everything goes on `/workspace` (persistent volume) so you don't lose it on pod restarts.

If the pod does not have a persistent `/workspace` volume, back up run artifacts before
terminating the pod:

```bash
bash scripts/backup_run_artifacts.sh /workspace/run_backups
```

Upload the produced tarball to a private Hugging Face dataset or another external store.
Do not back up `data/libero_long` by default; it is large and should be downloaded again
from `yifengzhu-hf/LIBERO-datasets`.

## Running Rollouts

```bash
# Single model, single task smoke test
bash libero_rollout_env/run_rollout.sh \
  configs/libero_long_sliding_window.yaml \
  checkpoints/libero_long/sliding_window/best.pt \
  --tasks 0 --episodes-per-task 1 --max-steps 50

# Full evaluation: all 4 models x 3 seeds x 10 tasks
bash libero_rollout_env/run_all_rollouts.sh

# Control episodes/steps via env vars
EPISODES_PER_TASK=20 MAX_STEPS=300 bash libero_rollout_env/run_all_rollouts.sh
```

## Environment Details

| Component | Location |
|-----------|----------|
| Isolated venv | `/workspace/libero_rollout_envs/.venv` |
| LIBERO source | `/workspace/libero_rollout_envs/LIBERO` |
| LIBERO data | `/workspace/vla-temporal-compression-baseline-data/libero_long` |
| Results CSV | `results/libero_rollouts.csv` |

## Recreate On A Fresh RunPod

```bash
cd /root/vla-temporal-compression-baseline
uv sync
export HF_HUB_ENABLE_HF_TRANSFER=1

bash libero_rollout_env/bootstrap.sh

HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir data/libero_long \
  --include "libero_10/*.hdf5" \
  --max-workers 2

uv run python scripts/inspect_libero.py --data-root data/libero_long
uv run python scripts/smoke_test.py --sources libero_long
```

If restoring a previous run, unpack the backup tarball from the repo root before running
eval or rollout.

## Current Checkpoint And Rollout State

As of 2026-05-25:

```text
sliding_window:
  checkpoint: checkpoints/libero_long_sliding_window_10ep_fixed/sliding_window/best.pt
  best epoch: 18
  best val_mse: 0.008474992022716574
  task-5 rollout: 0/1

event_gated_memory:
  checkpoint: checkpoints/libero_long/event_gated_memory/best.pt
  best epoch: 46
  best val_mse: 0.008947615628130734
  offline eval: MSE 0.06263986602425575, MAE 0.2735399380326271
  task-5 rollout: 0/1
  video: results/rollout_videos_event_gated_memory_50ep/event_gated_memory/seed42_task05_episode0_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4

age_gated_memory:
  checkpoint: checkpoints/libero_long/age_gated_memory/best.pt
  best epoch: 31
  best val_mse: 0.010890026518609375
  stopped at last.pt epoch 50 after the continuation completed
  rollout: pending
```

Run age-gated rollout after offline eval:

```bash
bash libero_rollout_env/run_rollout.sh   configs/ablation_gate_age.yaml   checkpoints/libero_long/age_gated_memory/best.pt   --tasks 5   --episodes-per-task 1   --max-steps 300   --video-dir results/rollout_videos_age_gated_memory_50ep   --video-every 1   --video-fps 20   --results-path results/libero_rollouts_age_gated_memory_50ep.csv
```

## Pinned Dependencies (known-working, from experimentation.md)

| Package | Version | Why pinned |
|---------|---------|------------|
| torch | 2.6.x + cu124 | robosuite compat, RTX 4090 |
| numpy | 1.26.4 | robosuite + mujoco need numpy <2 |
| mujoco | 3.8.1 | tested working version |
| robosuite | 1.4.0 | LIBERO requirement (not 1.4.1) |
| robomimic | 0.2.0 | LIBERO transitive dep |
| bddl | 1.0.1 | LIBERO requirement |
| gym | 0.25.2 | robosuite uses old gym API |

The main project uses torch 2.11+, numpy 2.x, gymnasium, etc. — those stay untouched.
