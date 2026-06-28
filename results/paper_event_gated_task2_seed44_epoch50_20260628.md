# Task-2 Event-Gated ACT Seed 44 Epoch-50 Result

Date: 2026-06-28

Task:

```text
KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
```

Training:

```text
config: configs/paper_event_gated_act_task2_seed44.yaml
checkpoint root: checkpoints/paper_event_gated_task2_seed44/event_gated_act_h20_task2_phase_memory
stopped after completed epoch: 50
best checkpoint by decoupled quick validation: best.pt, epoch 46
last checkpoint: last.pt, epoch 50
validation protocol note: val_split=train now uses deterministic eval windows instead of 20k stochastic train-mode samples.
```

Training-time validation:

```text
best.pt epoch: 46
best.pt val_mse: 0.044968954473733905
last.pt epoch: 50
last.pt val_mse: 0.10656119398772716
```

Offline eval:

| checkpoint | epoch | continuous_mse | continuous_mae | gripper_sign_accuracy |
|---|---:|---:|---:|---:|
| best.pt | 46 | 0.04581672772765159 | 0.14915308654308318 | 0.9846874952316285 |
| last.pt | 50 | 0.04505929201841354 | 0.14305126070976257 | 0.989312493801117 |

Epoch-50 rollout with temporal ensembling:

```text
checkpoint: checkpoints/paper_event_gated_task2_seed44/event_gated_act_h20_task2_phase_memory/last.pt
train30: 17/30
val5:     3/5
test5:    4/5
total:   24/40
held-out val+test: 7/10
```

Failed episodes:

```text
train: [0, 6, 8, 13, 16, 19, 21, 24, 25, 30, 34, 35, 36]
val:   [29, 41]
test:  [11]
```

Epoch-46 `best.pt` rollout with temporal ensembling:

```text
checkpoint: checkpoints/paper_event_gated_task2_seed44/event_gated_act_h20_task2_phase_memory/best.pt
train30: 17/30
val5:     2/5
test5:    2/5
total:   21/40
held-out val+test: 4/10
```

Epoch-46 failed episodes:

```text
train: [0, 1, 5, 8, 16, 17, 19, 25, 26, 33, 34, 36, 38]
val:   [9, 40, 41]
test:  [2, 11, 20]
```

Best-vs-last rollout flips:

```text
train last success -> best failure: [1, 5, 17, 26, 33, 38]
train best success -> last failure: [6, 13, 21, 24, 30, 35]
val last success -> best failure:   [9, 40]
val best success -> last failure:   [29]
test last success -> best failure:  [2, 20]
test best success -> last failure:  []
```

Artifacts:

```text
offline eval logs:
  logs/paper_event_gated_task2_seed44_eval_best_epoch46_20260628.log
  logs/paper_event_gated_task2_seed44_eval_last_epoch50_20260628.log

rollout CSVs:
  results/paper_rollouts_event_gated_task2_seed44_train30_epoch50.csv
  results/paper_rollouts_event_gated_task2_seed44_val5_epoch50.csv
  results/paper_rollouts_event_gated_task2_seed44_test5_epoch50.csv

trace CSVs:
  results/paper_trace_event_gated_task2_seed44_train30_epoch50.csv
  results/paper_trace_event_gated_task2_seed44_val5_epoch50.csv
  results/paper_trace_event_gated_task2_seed44_test5_epoch50.csv

artifact backup:
  https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/40def1523780664f7d84a1402c8294be0b8fdffa
```

Interpretation:

```text
Seed-44 event-gated epoch-50 reaches the same total 40-episode count as seed-43 event-gated
from scratch, but with a better held-out val+test count in this measured set.

seed-43 event-gated from scratch:
  train30 / val5 / test5 = 18/30, 3/5, 3/5 = 24/40
  held-out val+test = 6/10

seed-44 event-gated from scratch:
  train30 / val5 / test5 = 17/30, 3/5, 4/5 = 24/40
  held-out val+test = 7/10

Compared with phase ACT seed 44's earlier train10/val5/test5 result of 10/20, this seed-44
event-gated checkpoint is directionally stronger on the comparable aggregate:
train10 / val5 / test5 = 6/10, 3/5, 4/5 = 13/20.

This is not enough to claim event memory dominates phase ACT across seeds because phase ACT
seed 43 was 17/20 on the smaller protocol. It does support that the seed-44 event-gated
from-scratch run is viable and not a collapse.

The decoupled quick-validation best checkpoint did not select the better rollout controller:
epoch-46 `best.pt` ties epoch-50 `last.pt` on train30 but is worse on held-out val+test.
For seed-44 reporting, use epoch-50 `last.pt` unless a later full offline/rollout audit says otherwise.
```
