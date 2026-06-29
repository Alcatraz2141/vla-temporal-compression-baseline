# Task-2 Event-Gated ACT Seed 187 Epoch-50 Result

Date: 2026-06-29

Task:

```text
KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
```

Training:

```text
config: configs/paper_event_gated_act_task2_seed187.yaml
checkpoint root: checkpoints/paper_event_gated_task2_seed187/event_gated_act_h20_task2_phase_memory
stopped after completed epoch: 50
epoch-50 checkpoint: last.pt
best checkpoint by training validation: best.pt, epoch 39
training log: logs/paper_event_gated_task2_seed187_20260629_1101.log
epoch-50 train_loss: 0.028735
epoch-50 val_loss: 0.069750
best_val: 0.050476256757974625
```

The run was terminated after epoch 50 was fully checkpointed. Partial epoch-51 work was
discarded intentionally.

Offline eval:

| checkpoint | epoch | continuous_mse | continuous_mae | gripper_sign_accuracy |
|---|---:|---:|---:|---:|
| `last.pt` | 50 | 0.04214628413319588 | 0.13997110724449158 | 0.9911875009536744 |
| `best.pt` | 39 | 0.04945923686027527 | 0.15443556308746337 | 0.9894374966621399 |

Split-aware rollout with temporal ensembling on epoch-50 `last.pt`:

```text
max steps: 300
seed: 42
train30: 17/30
val5:     2/5
test5:    1/5
total:   20/40
held-out val+test: 3/10
```

Successful episodes:

```text
train: [0, 1, 15, 19, 21, 24, 25, 27, 28, 30, 32, 33, 34, 35, 36, 37, 38]
val:   [29, 40]
test:  [2]
```

Failed episodes:

```text
train: [4, 5, 6, 8, 10, 12, 13, 14, 16, 17, 18, 26, 31]
val:   [3, 9, 41]
test:  [7, 11, 20, 22]
```

Artifacts:

```text
offline eval logs:
  logs/paper_event_gated_task2_seed187_eval_last_epoch50_20260629.log
  logs/paper_event_gated_task2_seed187_eval_best_epoch39_20260629.log

rollout CSVs:
  results/paper_rollouts_event_gated_task2_seed187_train30_epoch50.csv
  results/paper_rollouts_event_gated_task2_seed187_val5_epoch50.csv
  results/paper_rollouts_event_gated_task2_seed187_test5_epoch50.csv

trace CSVs:
  results/paper_trace_event_gated_task2_seed187_train30_epoch50.csv
  results/paper_trace_event_gated_task2_seed187_val5_epoch50.csv
  results/paper_trace_event_gated_task2_seed187_test5_epoch50.csv
```

Interpretation:

```text
Seed-187 event-gated from scratch underperforms the matched seed-187 phase ACT rollout.

seed-187 phase ACT:
  train30 / val5 / test5 = 20/30, 1/5, 4/5 = 25/40
  held-out val+test = 5/10

seed-187 event-gated epoch-50:
  train30 / val5 / test5 = 17/30, 2/5, 1/5 = 20/40
  held-out val+test = 3/10

The event-gated model improves validation-split rollout relative to phase ACT but loses
substantially on test and total count. Together with seeds 43 and 44, the matched from-scratch
task-2 event-gated protocol does not support a broad claim that this event memory variant
dominates phase ACT. It does reinforce that offline metrics and quick validation are weak
closed-loop checkpoint selectors.
```
