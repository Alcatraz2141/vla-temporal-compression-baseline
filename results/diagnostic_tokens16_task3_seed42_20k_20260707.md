# Task-3 Seed-42 Tokens16 20k Reproduction Audit

Date: 2026-07-07

## Protocol

```text
task: KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
seed: 42
samples_per_epoch: 20000
epochs: 20
batch_size: 32
chunk_size: 4
max_memory_tokens: 16
older context frames: 64
rollout: split-aware train10 / val4 / test5, max_steps 300, temporal ensemble
```

Configs:

```text
configs/diagnostic_event_gated_act_task3_seed42_tokens16_20k.yaml
configs/diagnostic_age_gated_act_task3_seed42_tokens16_20k.yaml
```

Artifact backup:

```text
local: /workspace/run_backups/vla_task3_seed42_tokens16_20k_20260707.tar.gz
Hugging Face commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/3b6d6ce8e35a7950a9b3fe1b3a952f13be809193
```

## Training

Event-gated ACT:

```text
log: logs/diagnostic_event_gated_task3_seed42_tokens16_20k_20260707.log
checkpoint dir: checkpoints/diagnostic_event_gated_task3_seed42_tokens16_20k
best epoch by current decoupled validation: 17
best_val: 0.06795417641599973
epoch 20 train_loss: 0.044135
epoch 20 val_loss: 0.090545
```

Age-gated ACT:

```text
log: logs/diagnostic_age_gated_task3_seed42_tokens16_20k_20260707.log
checkpoint dir: checkpoints/diagnostic_age_gated_task3_seed42_tokens16_20k
best epoch by current decoupled validation: 20
best_val: 0.057931252444783844
epoch 20 train_loss: 0.041473
epoch 20 val_loss: 0.057931
```

## Offline Eval

```text
event-gated epoch-17 best.pt:
  continuous_mse: 0.09638847038149834
  continuous_mae: 0.20980131377776465
  gripper_sign_accuracy: 0.9846354226271311

event-gated epoch-20 last.pt:
  continuous_mse: 0.0882494921485583
  continuous_mae: 0.20001773784557977
  gripper_sign_accuracy: 0.9885416726271311

age-gated epoch-20 best.pt / last.pt:
  continuous_mse: 0.07588838351269563
  continuous_mae: 0.18988387286663055
  gripper_sign_accuracy: 0.9783854186534882
```

Offline metrics are current July eval-loader metrics. They should not be compared directly against
the old June offline CSV values without recreating the old train-mode validation/eval path.

## Rollouts

```text
event-gated epoch-17 best.pt:
  train10: 1/10, successes [9]
  val4:    2/4,  successes [1, 39]
  test5:   3/5,  successes [10, 35, 37]
  total:   6/19
  held-out val+test: 5/9

event-gated epoch-20 last.pt:
  train10: 7/10, successes [0, 2, 3, 4, 6, 9, 11]
  val4:    4/4,  successes [1, 8, 20, 39]
  test5:   4/5,  successes [10, 15, 35, 37]
  total:   15/19
  held-out val+test: 8/9

age-gated epoch-20 best.pt / last.pt:
  train10: 3/10, successes [0, 7, 11]
  val4:    1/4,  successes [39]
  test5:   1/5,  successes [35]
  total:   5/19
  held-out val+test: 2/9
```

Rollout CSVs:

```text
results/diagnostic_rollouts_event_gated_task3_seed42_tokens16_20k_train10_epoch17_best_20260707.csv
results/diagnostic_rollouts_event_gated_task3_seed42_tokens16_20k_val5_epoch17_best_20260707.csv
results/diagnostic_rollouts_event_gated_task3_seed42_tokens16_20k_test5_epoch17_best_20260707.csv
results/diagnostic_rollouts_event_gated_task3_seed42_tokens16_20k_train10_epoch20_last_20260707.csv
results/diagnostic_rollouts_event_gated_task3_seed42_tokens16_20k_val5_epoch20_last_20260707.csv
results/diagnostic_rollouts_event_gated_task3_seed42_tokens16_20k_test5_epoch20_last_20260707.csv
results/diagnostic_rollouts_age_gated_task3_seed42_tokens16_20k_train10_epoch20_best_20260707.csv
results/diagnostic_rollouts_age_gated_task3_seed42_tokens16_20k_val5_epoch20_best_20260707.csv
results/diagnostic_rollouts_age_gated_task3_seed42_tokens16_20k_test5_epoch20_best_20260707.csv
```

## Interpretation

The old Task-3 seed-42/20k/tokens16 rollout result is reproduced when event-gated reporting uses
epoch-20 `last.pt` rather than the current validation-selected epoch-17 `best.pt`.

The apparent mismatch was a checkpoint-selection artifact introduced by validation decoupling.
Current `build_dataloader` uses call intent rather than split name, so `val_split: train` no longer
runs the old stochastic train-mode validation path. This affects `best.pt` selection and offline
eval values, not the closed-loop rollout result from epoch-20 `last.pt`.

For Task-3 seed-42/20k/tokens16 reporting:

```text
event-gated: use epoch-20 last.pt, 15/19 total, 8/9 held-out
age-gated:   use epoch-20 best.pt / last.pt, 5/19 total, 2/9 held-out
```
