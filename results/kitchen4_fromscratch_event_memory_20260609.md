# Kitchen4 From-Scratch Event-Memory Probe

Date: 2026-06-09

Task:

```text
KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
LIBERO-10 task id: 3
```

Protocol:

```text
phase ACT from scratch, 20 epochs
event-gated ACT from scratch, 20 epochs
age-gated ACT from scratch, 20 epochs
same task split files
same task-3 rollout settings
temporal ensembling enabled
max rollout steps: 300
seed: 42
```

Offline eval:

| model | continuous_mse | continuous_mae | gripper_sign_accuracy |
|---|---:|---:|---:|
| phase ACT | 0.0519124200 | 0.1628629440 | 0.9896825003 |
| event-gated ACT | 0.0440423427 | 0.1500428320 | 0.9917249993 |
| age-gated ACT | 0.0470483830 | 0.1567056435 | 0.9881600015 |

Rollouts:

| split | phase ACT | event-gated ACT | age-gated ACT |
|---|---:|---:|---:|
| train | 3/10 | 7/10 | 3/10 |
| val | 1/4 | 4/4 | 1/4 |
| test | 2/5 | 4/5 | 1/5 |
| total | 6/19 | 15/19 | 5/19 |
| held-out val+test | 3/9 | 8/9 | 2/9 |

Matched rollout flips:

```text
phase failure -> event success:
  train: [2, 3, 6, 9]
  val:   [1, 8, 20]
  test:  [15, 37]

phase success -> event failure:
  none

event success -> age failure:
  train: [2, 3, 4, 6, 9]
  val:   [1, 8, 20]
  test:  [10, 15, 37]

phase failure -> age success:
  train: [7]
  val:   []
  test:  []
```

Interpretation:

```text
This is a strong positive repeat for event-gated memory under a from-scratch setup.
The offline improvement is moderate, while the closed-loop improvement is large.
The result is not a warm-start artifact because both phase ACT and event-gated ACT were trained from scratch.
The age-gated control improves offline MSE over phase ACT but does not reproduce the closed-loop gain.
This supports event-aware memory selection over simple recency/age selection on this task.
The main caveats are single seed and small rollout sets.
```

Relevant artifacts:

```text
configs/libero_long_act_chunked_h20_kitchen4_drawer_phase_fromscratch.yaml
configs/libero_long_event_gated_act_h20_kitchen4_drawer_phase_memory_fromscratch.yaml
configs/libero_long_age_gated_act_h20_kitchen4_drawer_phase_memory_fromscratch.yaml
checkpoints/libero_long_fromscratch_probe/act_chunked_h20_kitchen4_drawer_phase_fromscratch/best.pt
checkpoints/libero_long_fromscratch_probe/event_gated_act_h20_kitchen4_drawer_phase_memory_fromscratch/best.pt
checkpoints/libero_long_fromscratch_probe/age_gated_act_h20_kitchen4_drawer_phase_memory_fromscratch/best.pt
results/baselines_fromscratch_probe.csv
results/libero_rollouts_kitchen4_drawer_phase_fromscratch_train10.csv
results/libero_rollouts_kitchen4_drawer_phase_fromscratch_val5.csv
results/libero_rollouts_kitchen4_drawer_phase_fromscratch_test5.csv
results/libero_rollouts_kitchen4_drawer_event_gated_fromscratch_train10.csv
results/libero_rollouts_kitchen4_drawer_event_gated_fromscratch_val5.csv
results/libero_rollouts_kitchen4_drawer_event_gated_fromscratch_test5.csv
results/libero_rollouts_kitchen4_drawer_age_gated_fromscratch_train10.csv
results/libero_rollouts_kitchen4_drawer_age_gated_fromscratch_val5.csv
results/libero_rollouts_kitchen4_drawer_age_gated_fromscratch_test5.csv
```

Artifact backup:

```text
local: /workspace/run_backups/vla_run_artifacts_20260609_113957.tar.gz
Hugging Face dataset: Alcatraz1412/vla-run-backups
HF commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/cf4a2a58dce40859cf701b91da349781b25c44d6

latest local: /workspace/run_backups/vla_run_artifacts_20260609_180542.tar.gz
latest HF commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/a5c98d68099f2f050941034b3737b21cd3ed5875
```
