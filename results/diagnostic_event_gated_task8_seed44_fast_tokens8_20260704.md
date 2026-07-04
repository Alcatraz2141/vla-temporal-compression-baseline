# Task 8 Event-Gated Fast-Token Diagnostic, Seed 44

Date: 2026-07-04

Purpose: run a new-task event-gated diagnostic after the task-2 fast-token result showed age-gated strongly outperforming event-gated online.

Task:

```text
task id for rollout: 8
task filter: KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
```

Config:

```text
configs/diagnostic_event_gated_act_task8_seed44_fast_tokens8.yaml
```

Protocol:

```text
seed: 44
samples_per_epoch: 5000
epochs: 30
batch_size: 32
chunk_size: 4
max_memory_tokens: 8
older context frames: 32
cache_max_episodes: 128
num_workers: 8
prefetch_factor: 2
```

Rationale:

`max_memory_tokens=8` may make the event-gated diagnostic harder because it reduces older-context samples from 64 to 32. However, the task-2 age-gated run used the same 8-token budget and reached 19/40 total with 6/10 held-out, while the task-2 event-gated run reached only 2/40 and 0/10 held-out. That matched result means compression alone does not explain the event-gated failure.

This task-8 run tests whether the current event gate fails broadly under the fast protocol or whether task 2 is unusually hostile to the event gate.

Status:

```text
event-gated and matched age-gated diagnostics completed
```

## Training

Event-gated:

```text
config: configs/diagnostic_event_gated_act_task8_seed44_fast_tokens8.yaml
log: logs/diagnostic_event_gated_task8_seed44_fast_tokens8_20260704.log
completed epoch: 30
best epoch: 27
best val_loss: 0.17377837151288986
epoch-30 val_loss: 0.18499184846878053
checkpoint: checkpoints/diagnostic_event_gated_task8_seed44_fast_tokens8/diagnostic_event_gated_act_h20_task8_fast_tokens8_seed44/best.pt
```

Age-gated:

```text
config: configs/diagnostic_age_gated_act_task8_seed44_fast_tokens8.yaml
log: logs/diagnostic_age_gated_task8_seed44_fast_tokens8_20260704.log
completed epoch: 30
best epoch: 29
best val_loss: 0.14792844355106355
epoch-30 val_loss: 0.1890100955963135
checkpoint: checkpoints/diagnostic_age_gated_task8_seed44_fast_tokens8/diagnostic_age_gated_act_h20_task8_fast_tokens8_seed44/best.pt
```

## Offline Eval

Event-gated best epoch 27:

```text
offline continuous_mse: 0.15478821992874145
offline continuous_mae: 0.28287607729434966
gripper_sign_accuracy: 0.9620535731315613
```

Age-gated best epoch 29:

```text
offline continuous_mse: 0.13650239408016204
offline continuous_mae: 0.2640184283256531
gripper_sign_accuracy: 0.9669196486473084
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
event-gated epoch-27 best:
  val:  0/5
  test: 0/2
  held-out: 0/7
  failures: val [0, 32, 37, 39, 40], test [29, 30]

age-gated epoch-29 best:
  val:  0/5
  test: 0/2
  held-out: 0/7
  failures: val [0, 32, 37, 39, 40], test [29, 30]
```

Rollout files:

```text
results/diagnostic_rollouts_event_gated_task8_seed44_fast_tokens8_val5_epoch27_best_20260704.csv
results/diagnostic_rollouts_event_gated_task8_seed44_fast_tokens8_test5_epoch27_best_20260704.csv
results/diagnostic_rollouts_age_gated_task8_seed44_fast_tokens8_val5_epoch29_best_20260704.csv
results/diagnostic_rollouts_age_gated_task8_seed44_fast_tokens8_test5_epoch29_best_20260704.csv
```

## Readout

Both task-8 fast-token memory variants failed held-out rollout completely. Age-gated is stronger offline, but neither checkpoint is rollout-competent under this 30-epoch, 5000-samples-per-epoch, 8-token diagnostic protocol.

This result does not rescue the current event gate. Across fast diagnostics:

```text
task 2: age-gated much stronger online than event-gated
task 8: both fail held-out
task 3: earlier event-gated result was strong
```

The current evidence suggests event-gated behavior is task-sensitive and unreliable. The next useful step is likely gate-design work or a less-compressed diagnostic (`max_memory_tokens=16`) on task 8, not more 8-token task sweeps.

## Tokens16 Follow-Up

Task-8 event-gated was rerun with `max_memory_tokens=16` and the same 5000-samples-per-epoch,
30-epoch diagnostic budget.

```text
summary: results/diagnostic_event_gated_task8_seed44_tokens16_20260704.md
best epoch: 30
offline continuous_mse: 0.14378715455532073
rollout val/test: 0/5, 0/2 = 0/7 held-out
```

Readout:

```text
Increasing memory tokens improved offline action prediction but did not recover held-out rollout.
Task-8 remains nonfunctional online for event-gated under both 8-token and 16-token diagnostics.
```
