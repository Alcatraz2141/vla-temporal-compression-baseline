# Task 2 Continued Phase ACT Control

Date: 2026-06-08

Purpose:

```text
Control for the event-gated ACT warm-start confound.
Compare event-gated ACT against a phase ACT model that receives the same continued training budget
from the task-2 phase ACT checkpoint, without adding memory parameters.
```

Task:

```text
KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
```

Config:

```text
configs/libero_long_act_chunked_corrected_h20_task2_phase_continued20.yaml
```

Checkpoint:

```text
checkpoints/libero_long_corrected_task2/act_chunked_corrected_h20_task2_phase_continued20/best.pt
```

Training:

```text
resume source: checkpoints/libero_long_corrected_task2/act_chunked_corrected_h20_task2_phase_conditioned/best.pt
resume source epoch: 29
continued epochs: 30..49
best epoch: 49
best val_loss: 0.022997978460788725
```

Offline eval:

| Model | continuous_mse | continuous_mae | gripper_sign_accuracy | pred_temporal_smoothness |
|---|---:|---:|---:|---:|
| Phase ACT original | 0.033583 | 0.132092 | 0.984755 | 0.164605 |
| Phase ACT continued20 | 0.023285 | 0.109139 | 0.990700 | 0.297369 |
| Event-gated ACT | 0.022641 | 0.107986 | 0.989935 | 0.242632 |

Rollout result:

| Split | Phase ACT original | Phase ACT continued20 | Event-gated ACT |
|---|---:|---:|---:|
| train10 | 9/10 | 6/10 | 10/10 |
| val5 | 2/5 | 3/5 | 5/5 |
| test5 | 4/5 | 3/5 | 4/5 |
| total | 15/20 | 12/20 | 19/20 |
| held-out val+test | 6/10 | 6/10 | 9/10 |

Episode-level comparison:

```text
continued phase ACT fixes:
  train ep6
  val ep29
  test ep7

continued phase ACT newly breaks relative to original phase:
  train ep0
  train ep4
  train ep10
  train ep12
  test ep11
  test ep22

event-gated ACT remains stronger overall:
  train: fixes all original/continued phase failures
  val: fixes ep29, ep40, ep41
  test: succeeds on ep7, ep11, ep22 but fails ep20
```

Interpretation:

```text
The control is important. Continued phase ACT receives the same extra training budget and nearly
matches event-gated ACT offline, but it does not match event-gated ACT in closed-loop rollout.

This means the task-2 event-gated improvement is not explained by "just train the phase model
longer." Extra training alone improves offline error but hurts aggregate rollout success
from 15/20 to 12/20. Event-gated ACT reaches 19/20 under the same selected rollout protocol.

The strongest defensible claim after this control is:
event-gated memory improves task-2 closed-loop robustness beyond an equal-budget continued
phase-ACT control, even though offline metrics alone would not reveal the gap.
```

Artifacts:

```text
logs/act_chunked_corrected_h20_task2_phase_continued20_20260608.log
results/baselines_corrected_task2.csv
results/libero_rollouts_phase_act_task2_continued20_train10.csv
results/libero_rollouts_phase_act_task2_continued20_val5.csv
results/libero_rollouts_phase_act_task2_continued20_test5.csv
results/rollout_trace_phase_act_task2_continued20_train10.csv
results/rollout_trace_phase_act_task2_continued20_val5.csv
results/rollout_trace_phase_act_task2_continued20_test5.csv
```
