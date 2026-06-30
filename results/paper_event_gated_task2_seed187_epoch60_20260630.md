# Task-2 Event-Gated ACT Seed 187 Epoch-60 Audit

Date: 2026-06-30

```text
config: configs/paper_event_gated_act_task2_seed187_resume.yaml
checkpoint root: checkpoints/paper_event_gated_task2_seed187/event_gated_act_h20_task2_phase_memory
last.pt epoch: 60
best.pt epoch: 56
best_val: 0.03878678865730763
epoch-60 val_mse: 0.06813872456550599
```

Offline eval:

```text
epoch-60 last.pt continuous_mse: 0.03697693571448326
epoch-60 last.pt continuous_mae: 0.13206598460674285
epoch-60 last.pt gripper_sign_accuracy: 0.987250006198883

epoch-56 best.pt continuous_mse: 0.036199239641427995
epoch-56 best.pt continuous_mae: 0.1305200546979904
epoch-56 best.pt gripper_sign_accuracy: 0.9880625009536743
```

Rollout on epoch-60 `last.pt`:

```text
suite: LIBERO-10
task: KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
max_steps: 300
temporal_ensemble: true
rollout_seed: 42
split protocol: train30 / val5 / test5
```

```text
train30: 13/30
val5: 2/5
test5: 2/5
total: 17/40
held-out val+test: 4/10
```

Failure episode IDs:

```text
train failures: [4, 5, 8, 10, 13, 15, 16, 17, 18, 24, 25, 26, 27, 33, 35, 37, 38]
val failures: [3, 9, 40]
test failures: [2, 7, 20]
```

Artifacts:

```text
training log:
  logs/paper_event_gated_task2_seed187_resume_51_60_20260630.log
offline eval logs:
  logs/paper_event_gated_task2_seed187_eval_last_epoch60_20260630.log
  logs/paper_event_gated_task2_seed187_eval_best_epoch56_20260630.log
rollout logs:
  logs/paper_event_gated_task2_seed187_rollout_train30_epoch60_20260630.log
  logs/paper_event_gated_task2_seed187_rollout_val5_epoch60_20260630.log
  logs/paper_event_gated_task2_seed187_rollout_test5_epoch60_20260630.log
rollout CSVs:
  results/paper_rollouts_event_gated_task2_seed187_train30_epoch60.csv
  results/paper_rollouts_event_gated_task2_seed187_val5_epoch60.csv
  results/paper_rollouts_event_gated_task2_seed187_test5_epoch60.csv
```

Comparison:

```text
event-gated seed 187 epoch 50:
  offline continuous_mse: 0.04214628413319588
  rollout train30 / val5 / test5: 17/30, 2/5, 1/5 = 20/40
  held-out val+test: 3/10

event-gated seed 187 epoch 60:
  offline continuous_mse: 0.03697693571448326
  rollout train30 / val5 / test5: 13/30, 2/5, 2/5 = 17/40
  held-out val+test: 4/10

cold-start phase ACT seed 187:
  offline continuous_mse: 0.03023523513815905
  rollout train30 / val5 / test5: 25/30, 4/5, 4/5 = 33/40
  held-out val+test: 8/10
```

Interpretation:

```text
Continuing event-gated seed 187 from epoch 50 to 60 improved offline action prediction but
worsened total rollout from 20/40 to 17/40. Held-out rollout increased slightly from 3/10 to
4/10, but remains far below cold-start phase ACT seed 187 at 8/10. This reinforces the current
finding that offline MSE is not a reliable closed-loop checkpoint selector and that the matched
task-2 event-gated ACT result is negative relative to cold-start phase ACT.
```
