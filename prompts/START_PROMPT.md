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

The corrected-H1 path has now produced real online success on a task-5 overfit run, so the rollout stack is not fundamentally broken. The active problem is full multitask reliability, especially rare exact gripper transition timing.

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

June 1/2 corrected-H1 results:

```text
Task-5 transition-aware overfit:
  checkpoint: checkpoints/libero_long_corrected_task5/sliding_window_corrected_h1_task5_overfit/best.pt
  task-5 train split rollout: 5/5
  task-5 val split rollout: 2/5
  task-5 test split rollout: 5/5

Full multitask sliding-window transition20:
  checkpoint: checkpoints/libero_long_corrected_transition20/sliding_window_corrected_h1_transition20/best.pt
  best epoch: 13
  best val_loss: 0.05925493258982897
  eval continuous_mse: 0.0438247038051486
  eval continuous_mae: 0.11593457460403442
  eval gripper_sign_accuracy: 0.9708333373069763
  quick train-init rollouts task 0/2/5: 0/3, 0/3, 0/3

Per-task validation transition diagnostic:
  results/per_task_transition_diagnostics_transition20_val.csv
  mean task continuous_mse: 0.000862
  mean task continuous_mae: 0.017611
  mean task gripper accuracy: 0.966178
  overall exact transition accuracy: 101/175 = 0.577143
  overall near-transition accuracy: 943/1216 = 0.775493

Task-5 comparison:
  full_transition20 transition hits: 8/15
  task5_overfit transition hits: 12/15
```

Interpretation:

```text
Average continuous action prediction is good.
Overall gripper sign accuracy is misleading because non-transition frames dominate.
The full multitask model still misses too many rare exact gripper transition frames.
The paper idea is not abandoned; we need a competent reactive baseline before memory comparisons are interpretable.
```

## Immediate RunPod Work

Do not launch event-gated memory as the immediate next run.

First improve the full multitask sliding-window protocol:

```text
1. Strengthen task-balanced plus transition-balanced sampling.
2. Track per-task exact transition accuracy as a primary metric.
3. If transition accuracy remains weak, add stronger task conditioning.
4. Only then train event-gated memory with the same improved protocol.
```

Start by inspecting and editing the data loader/training config around:

```text
datasets/episode_loader.py
datasets/data_loader.py
training/train.py
configs/libero_long_sliding_window_corrected_h1.yaml
configs/libero_long_event_gated_corrected_h1.yaml
```

The next training run should answer:

```text
Can the full multitask reactive baseline learn reliable transition timing?
```

Suggested target:

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

# After implementing/configuring stronger task-balanced + transition-balanced sampling:
uv run python train.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml \
  --epochs 20

uv run python evaluation/eval.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml

uv run python evaluation/offline_diagnostics.py \
  --config configs/libero_long_sliding_window_corrected_h1.yaml \
  --checkpoint checkpoints/libero_long_corrected/sliding_window_corrected_h1/best.pt
```

Then rerun per-task transition diagnostics and task 0/2/5 train-init rollouts.

## Decision Rule

Continue scaling only if:

```text
per-task exact transition accuracy improves materially beyond 101/175
weak tasks improve, especially living-room multi-object tasks
continuous_mse/continuous_mae do not regress badly
task 0/2/5 train-init rollouts become nonzero or qualitatively closer
no NaN/inf diagnostics
no OOM or dataloader instability
```

If the improved sliding-window protocol passes, run event-gated corrected-H1 with the same protocol:

```bash
uv run python train.py \
  --config configs/libero_long_event_gated_corrected_h1.yaml \
  --epochs 20

uv run python evaluation/eval.py \
  --config configs/libero_long_event_gated_corrected_h1.yaml

uv run python evaluation/offline_diagnostics.py \
  --config configs/libero_long_event_gated_corrected_h1.yaml \
  --checkpoint checkpoints/libero_long_corrected/event_gated_memory_corrected_h1/best.pt
```

Only run broad LIBERO simulator rollout after offline transition diagnostics are sane.
Start with task 0/2/5 training-init rollouts, then use held-out split rollouts only if training-init behavior is credible.

## Do Not

- Do not retrain the old H=4 configs as the next step.
- Do not run broad memory sweeps before the corrected sliding-window transition problem improves.
- Do not claim rollout success from offline metrics.
- Do not compare old pre-fix/pre-H1 MSE directly against corrected-H1 MSE.
- Do not upload LIBERO data to the user Hugging Face repo.
- Do not skip artifact backups before terminating a non-persistent pod.

Before starting expensive work, reply with a short summary of what you understood and the exact first command you will run.
