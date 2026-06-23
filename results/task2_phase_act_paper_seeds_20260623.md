# Task-2 Phase-ACT Paper Seeds

Date: 2026-06-23

## Protocol

```text
task id: 2
task: KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
model: phase-conditioned ACT, H=20
rollout protocol: train10 / val5 / test5
max steps: 300
simulator seed: 42
```

## Seed 43

```text
checkpoint root: checkpoints/paper_phase_act_task2_seed43
best checkpoint: checkpoints/paper_phase_act_task2_seed43/act_chunked_corrected_h20_task2_phase_conditioned/best.pt
best epoch: 57
best_val: 0.01841944182217121
last epoch: 60
last val_mse: 0.021169397957623005

offline eval:
  continuous_mse: 0.01966886719018221
  continuous_mae: 0.10089445358514786
  gripper_sign_accuracy: 0.9917099995613098

rollouts:
  train10: 8/10
  val5:    4/5
  test5:   5/5
  total:   17/20
```

Successful episodes:

```text
train: [0, 1, 5, 6, 8, 12, 13, 14]
val:   [3, 9, 29, 40]
test:  [2, 7, 11, 20, 22]
```

Failed episodes:

```text
train: [4, 10]
val:   [41]
test:  []
```

## Seed 44

```text
checkpoint root: checkpoints/paper_phase_act_task2_seed44
resume config: configs/paper_phase_act_task2_seed44_resume.yaml
interrupted checkpoint before recovery: last.pt epoch 13
completed epoch: 60
best checkpoint: checkpoints/paper_phase_act_task2_seed44/act_chunked_corrected_h20_task2_phase_conditioned/best.pt
best epoch: 58
best_val: 0.018637190474569798
last epoch: 60
last val_mse: 0.021783009807765485

offline eval:
  continuous_mse: 0.020462753280997278
  continuous_mae: 0.10250961701869965
  gripper_sign_accuracy: 0.9940675008773804

rollouts:
  train10: 4/10
  val5:    3/5
  test5:   3/5
  total:   10/20
```

Successful episodes:

```text
train: [0, 6, 12, 14]
val:   [3, 29, 41]
test:  [7, 20, 22]
```

Failed episodes:

```text
train: [1, 4, 5, 8, 10, 13]
val:   [9, 40]
test:  [2, 11]
```

## Matched Comparison

```text
offline continuous_mse:
  seed 43: 0.01966886719018221
  seed 44: 0.020462753280997278

rollout total:
  seed 43: 17/20
  seed 44: 10/20

held-out val+test:
  seed 43: 9/10
  seed 44: 6/10
```

Interpretation:

```text
Seed 43 is a strong phase-ACT task-2 controller.
Seed 44 has similar offline metrics but much weaker closed-loop rollout.
The result is seed sensitivity in online control, not a clean offline-metric story.
Do not use this as evidence that longer phase-ACT training alone reliably closes the event-memory gap.
```

## Left To Do

```text
1. Train event-gated ACT task-2 phase-memory from the seed-43 phase checkpoint.
2. Train event-gated ACT task-2 phase-memory from the seed-44 phase checkpoint.
3. Run the same train10 / val5 / test5 rollouts.
4. Compare per-episode flips and held-out aggregate success.
5. Use those matched seeds for the next paper-level claim.
```

## Artifact Files

```text
logs/paper_phase_act_task2_seed43_20260623_1308.log
logs/paper_phase_act_task2_seed43_eval_20260623.log
logs/paper_phase_act_task2_seed43_rollout_train10_20260623.log
logs/paper_phase_act_task2_seed43_rollout_val5_20260623.log
logs/paper_phase_act_task2_seed43_rollout_test5_20260623.log
logs/paper_phase_act_task2_seed44_train_20260623.log
logs/paper_phase_act_task2_seed44_resume_monitor_20260623.log
logs/paper_phase_act_task2_seed44_resume_20260623.log
logs/paper_phase_act_task2_seed44_eval_20260623.log
logs/paper_phase_act_task2_seed44_rollout_train10_20260623.log
logs/paper_phase_act_task2_seed44_rollout_val5_20260623.log
logs/paper_phase_act_task2_seed44_rollout_test5_20260623.log
results/paper_baselines_phase_act_task2_seed44.csv
results/paper_rollouts_phase_act_task2_seed43_train10.csv
results/paper_rollouts_phase_act_task2_seed43_val5.csv
results/paper_rollouts_phase_act_task2_seed43_test5.csv
results/paper_rollouts_phase_act_task2_seed44_train10.csv
results/paper_rollouts_phase_act_task2_seed44_val5.csv
results/paper_rollouts_phase_act_task2_seed44_test5.csv
results/paper_trace_phase_act_task2_seed43_train10.csv
results/paper_trace_phase_act_task2_seed43_val5.csv
results/paper_trace_phase_act_task2_seed43_test5.csv
results/paper_trace_phase_act_task2_seed44_train10.csv
results/paper_trace_phase_act_task2_seed44_val5.csv
results/paper_trace_phase_act_task2_seed44_test5.csv
```

Backup:

```text
local: /workspace/run_backups/vla_run_artifacts_20260623_191057.tar.gz
Hugging Face commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/b415453cb949fd7cd6ebb2ba8abdae9a2c0ed72b
```
