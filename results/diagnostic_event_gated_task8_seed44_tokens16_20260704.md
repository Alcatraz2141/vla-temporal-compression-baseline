# Task 8 Event-Gated Tokens16 Diagnostic, Seed 44

Date: 2026-07-04

Purpose: test whether the task-8 event-gated failure under the fast `max_memory_tokens=8` protocol was caused by over-compressing older context.

Task:

```text
task id for rollout: 8
task filter: KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
```

Config:

```text
configs/diagnostic_event_gated_act_task8_seed44_tokens16.yaml
```

Protocol:

```text
seed: 44
samples_per_epoch: 5000
epochs: 30
batch_size: 32
chunk_size: 4
max_memory_tokens: 16
older context frames: 64
cache_max_episodes: 128
num_workers: 8
prefetch_factor: 2
```

Status:

```text
completed
```

## Training

```text
log: logs/diagnostic_event_gated_task8_seed44_tokens16_20260704.log
completed epoch: 30
best epoch: 30
best val_loss: 0.16754465252161027
checkpoint: checkpoints/diagnostic_event_gated_task8_seed44_tokens16/diagnostic_event_gated_act_h20_task8_tokens16_seed44/best.pt
```

Compared with task-8 event-gated tokens8:

```text
tokens8 best val_loss:  0.17377837151288986
tokens16 best val_loss: 0.16754465252161027
```

## Offline Eval

```text
offline continuous_mse: 0.14378715455532073
offline continuous_mae: 0.26980500519275663
gripper_sign_accuracy: 0.9627232193946839
pred_temporal_smoothness: 0.03873096257448196
```

Compared with task-8 tokens8:

```text
event tokens8 continuous_mse: 0.15478821992874145
event tokens16 continuous_mse: 0.14378715455532073
age tokens8 continuous_mse:   0.13650239408016204
```

## Held-Out Rollout

Rollout protocol:

```text
task: 8
max_steps: 300
temporal_ensemble: true
seed: 42
splits: val5 / test split-aware task-8 episodes
```

The split-aware test file yielded only two task-8 episodes, so the held-out total is 7 episodes.

```text
event-gated tokens16 epoch-30 best:
  val:  0/5
  test: 0/2
  held-out: 0/7
  failures: val [0, 32, 37, 39, 40], test [29, 30]
```

Rollout files:

```text
results/diagnostic_rollouts_event_gated_task8_seed44_tokens16_val5_epoch30_best_20260704.csv
results/diagnostic_rollouts_event_gated_task8_seed44_tokens16_test5_epoch30_best_20260704.csv
```

## Readout

Increasing `max_memory_tokens` from 8 to 16 improved task-8 event-gated offline prediction but did
not recover held-out rollout success. Task-8 remains 0/7 held-out for event-gated at both memory
budgets tested.

Current task-8 diagnostic comparison:

```text
event tokens8:  continuous_mse 0.154788, held-out 0/7
event tokens16: continuous_mse 0.143787, held-out 0/7
age tokens8:    continuous_mse 0.136502, held-out 0/7
```

Conclusion: the task-8 failure is not simply caused by the 8-token compression setting. Larger
memory helps offline but still does not produce closed-loop competence under this 30-epoch,
5000-samples-per-epoch protocol.
