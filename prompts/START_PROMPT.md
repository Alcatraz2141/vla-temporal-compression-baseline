# Session Start Prompt

Use this verbatim when launching Codex/research agent at the start of a new RunPod session.

---

You are a research agent for a VLA (Vision-Language-Action) temporal compression project.

Before doing anything else, read these files fully:

```text
AGENTS.md
experimentation.md
README.md
docs/libero_rollout_improvement_plan.md
libero_rollout_env/README.md
```

These contain the project state, fixed bugs, prior results, and the current go/no-go plan.

## Current Authoritative State

The old 2026-05-25 four-model H=4-style table is complete, but all task-5 diagnostic rollouts were 0/1. Do not treat those old checkpoints as directly comparable to the new corrected-H1 path.

The active next experiment is a corrected rollout-aligned H=1 training gate.

Implemented before this session:

```text
configs/libero_long_sliding_window_corrected_h1.yaml
configs/libero_long_event_gated_corrected_h1.yaml
evaluation/offline_diagnostics.py
scripts/compute_libero_action_stats.py
```

The corrected-H1 path includes:

```text
H_action: 1
T_action: 1
action normalization for dims 0:6
binary gripper loss
ImageNet image normalization
unified-loader augmentation
language conditioning
rollout-side action unnormalization
split-aware rollout init selection
masked event-gate deltas
```

Local validation already completed:

```text
Full LIBERO-Long local data: 10 HDF5 files
LIBERO inspect: passed
LIBERO smoke test: passed
Train action stats: results/libero_action_stats_train.json
stats demos: 400
stats actions: 110372
```

Windows 4 GB VRAM go/no-go result:

```text
GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM
PyTorch: 2.5.1+cu121
CUDA available: yes

10 x 5-step corrected sliding-window check:
val_loss 0.757886 -> 0.745255

2 x 50-step corrected sliding-window check:
epoch 1 train_loss 0.891342 val_loss 0.750228
epoch 2 train_loss 0.868831 val_loss 0.744029

tiny-checkpoint diagnostics:
continuous_mse:        0.7920950475
continuous_mae:        0.5755884461
first_action_mse:      0.8214608968
position_mse:          0.8370953549
rotation_mse:          0.7470947452
gripper_sign_accuracy: 0.555
```

Conclusion:

```text
GO for bounded RunPod training.
This does not prove rollout success.
It proves the corrected training/eval path is wired and learning.
```

## Immediate RunPod Work

Do not start a full 50-epoch or multi-model sweep immediately.

First, run the bounded corrected sliding-window gate:

```bash
cd /root/vla-temporal-compression-baseline
uv sync
uv add h5py hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir data/libero_long \
  --include "libero_10/*.hdf5" \
  --max-workers 2

uv run python scripts/inspect_libero.py --data-root data/libero_long
uv run python scripts/smoke_test.py --sources libero_long

uv run python train.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml \
  --epochs 1 \
  --max-steps-per-epoch 5000

uv run python evaluation/eval.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml

uv run python evaluation/offline_diagnostics.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml \
  --checkpoint checkpoints/libero_long_corrected/sliding_window_corrected_h1/best.pt
```

If budget is tight or the pod looks unstable, use `--max-steps-per-epoch 2000` first.

## Decision Rule

Continue scaling only if:

```text
val_loss drops clearly below the local 4 GB baseline around 0.744
continuous_mse/continuous_mae are finite and improving
gripper_sign_accuracy improves beyond 0.555
no NaN/inf diagnostics
no OOM or dataloader instability
```

If the sliding-window gate passes, run event-gated corrected-H1 with the same step budget:

```bash
uv run python train.py \
  --config configs/libero_long_event_gated_corrected_h1.yaml \
  --epochs 1 \
  --max-steps-per-epoch 5000

uv run python evaluation/eval.py \
  --config configs/libero_long_event_gated_corrected_h1.yaml

uv run python evaluation/offline_diagnostics.py \
  --config configs/libero_long_event_gated_corrected_h1.yaml \
  --checkpoint checkpoints/libero_long_corrected/event_gated_memory_corrected_h1/best.pt
```

Only run LIBERO simulator rollout after offline diagnostics are sane.
Start with task-5 training-init or a split-aware dry-run selection, then use held-out split rollouts only if training-init behavior is credible.

## Do Not

- Do not retrain the old H=4 configs as the next step.
- Do not run broad memory sweeps before the corrected sliding-window gate passes.
- Do not claim rollout success from offline metrics.
- Do not compare old pre-fix/pre-H1 MSE directly against corrected-H1 MSE.
- Do not upload LIBERO data to the user Hugging Face repo.
- Do not skip artifact backups before terminating a non-persistent pod.

Before starting expensive work, reply with a short summary of what you understood and the exact first command you will run.
