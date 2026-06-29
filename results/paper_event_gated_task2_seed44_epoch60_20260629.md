# Task-2 Event-Gated ACT Seed 44 Epoch-60 Result

Date: 2026-06-29

Task:

```text
KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
```

Training:

```text
config: configs/paper_event_gated_act_task2_seed44_resume.yaml
checkpoint root: checkpoints/paper_event_gated_task2_seed44/event_gated_act_h20_task2_phase_memory
resume source: epoch-50 last.pt
completed epoch: 60
epoch-60 checkpoint: last.pt
training log: logs/paper_event_gated_task2_seed44_resume_50to60_20260629_0919.log
epoch-60 train_loss: 0.025051
epoch-60 val_loss: 0.053994
best checkpoint by training validation remains: best.pt, epoch 46
```

Offline eval on epoch-60 `last.pt`:

```text
continuous_mse: 0.04118259623646736
continuous_mae: 0.13802042752504348
gripper_sign_accuracy: 0.9870000004768371
pred_temporal_smoothness: 0.2989514470100403
```

For comparison, epoch-50 `last.pt` offline eval was:

```text
continuous_mse: 0.04505929201841354
continuous_mae: 0.14305126070976257
gripper_sign_accuracy: 0.989312493801117
```

Split-aware rollout with temporal ensembling:

```text
checkpoint: checkpoints/paper_event_gated_task2_seed44/event_gated_act_h20_task2_phase_memory/last.pt
max steps: 300
seed: 42
train30: 16/30
val5:     1/5
test5:    1/5
total:   18/40
held-out val+test: 2/10
```

Failed episodes:

```text
train: [0, 1, 5, 6, 12, 14, 16, 19, 24, 25, 26, 28, 34, 38]
val:   [9, 29, 40, 41]
test:  [7, 11, 20, 22]
```

Successful episodes:

```text
train: [4, 8, 10, 13, 15, 17, 18, 21, 27, 30, 31, 32, 33, 35, 36, 37]
val:   [3]
test:  [2]
```

Artifacts:

```text
offline eval log:
  logs/paper_event_gated_task2_seed44_eval_last_epoch60_20260629.log

rollout CSVs:
  results/paper_rollouts_event_gated_task2_seed44_train30_epoch60.csv
  results/paper_rollouts_event_gated_task2_seed44_val5_epoch60.csv
  results/paper_rollouts_event_gated_task2_seed44_test5_epoch60.csv

trace CSVs:
  results/paper_trace_event_gated_task2_seed44_train30_epoch60.csv
  results/paper_trace_event_gated_task2_seed44_val5_epoch60.csv
  results/paper_trace_event_gated_task2_seed44_test5_epoch60.csv
```

Interpretation:

```text
Continuing seed-44 event-gated ACT from epoch 50 to epoch 60 improved offline continuous MSE
and MAE, but materially worsened rollout success.

epoch-50 last.pt rollout:
  train30 / val5 / test5 = 17/30, 3/5, 4/5 = 24/40
  held-out val+test = 7/10

epoch-60 last.pt rollout:
  train30 / val5 / test5 = 16/30, 1/5, 1/5 = 18/40
  held-out val+test = 2/10

For seed-44 reporting, keep epoch-50 last.pt as the rollout-facing checkpoint unless a later
audit finds a stronger checkpoint. This is another example where improved offline action
prediction did not select the better closed-loop controller.
```
