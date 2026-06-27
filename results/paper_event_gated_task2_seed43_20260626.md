# Task-2 Event-Gated ACT Seed 43 From-Scratch Result

Date: 2026-06-26

## Run

```text
task: KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
seed: 43
config: configs/paper_event_gated_act_task2_seed43_resume.yaml
checkpoint dir: checkpoints/paper_event_gated_task2_seed43/event_gated_act_h20_task2_phase_memory
resume source: last.pt from epoch 21
completed epoch: 60
best checkpoint: best.pt from epoch 58
best val_mse: 0.020923468711972235
last epoch val_mse: 0.020978209085762502
```

Training was resumed at 2026-06-26 05:02:42 UTC and completed epoch 60 at
2026-06-26 16:16:08 UTC.

## Offline Eval

```text
checkpoint: checkpoints/paper_event_gated_task2_seed43/event_gated_act_h20_task2_phase_memory/best.pt
results: results/paper_baselines_event_gated_task2_seed43.csv
continuous_mse: 0.01848969299942255
continuous_mae: 0.09646275240182876
gripper_sign_accuracy: 0.9892899991989136
```

This is slightly better offline continuous MSE than the matched seed-43 phase-ACT checkpoint
recorded earlier (`0.01966886719018221`).

## Split-Aware Rollout

Protocol:

```text
split-aware unique task-2 rollout
train30 / val5 / test5
max steps: 300
seed: 42
temporal ensembling: enabled
videos: not recorded
```

Results:

```text
train30: 18/30
val5:     3/5
test5:    3/5
total:   24/40
held-out val+test: 6/10
```

Failed episodes:

```text
train: [0, 4, 5, 6, 15, 16, 19, 21, 26, 31, 34, 38]
val:   [29, 40]
test:  [7, 22]
```

Artifacts:

```text
results/paper_rollouts_event_gated_task2_seed43_train30.csv
results/paper_rollouts_event_gated_task2_seed43_val5.csv
results/paper_rollouts_event_gated_task2_seed43_test5.csv
results/paper_trace_event_gated_task2_seed43_train30.csv
results/paper_trace_event_gated_task2_seed43_val5.csv
results/paper_trace_event_gated_task2_seed43_test5.csv
backup: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/db21d6240870b8a95ecdc8b39337f8a2691faad4
```

## Interpretation

The from-scratch event-gated seed-43 model improved offline continuous action prediction but did
not reproduce the strong seed-43 phase-ACT online rollout result.

Comparable 20-episode subset:

```text
phase ACT seed 43:              train10 8/10, val5 4/5, test5 5/5 = 17/20
event-gated from scratch seed43: train10 6/10, val5 3/5, test5 3/5 = 12/20
```

The correct conclusion is that this matched from-scratch event-memory seed does not support a
simple "event memory improves seed 43" claim. It reinforces that offline MSE is insufficient for
selecting robust closed-loop checkpoints. Continue with matched from-scratch event-gated seeds 44
and 187 before deciding whether to run age-gated controls.
