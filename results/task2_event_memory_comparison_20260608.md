# Task 2 Phase ACT vs Event-Gated Memory ACT

Date: 2026-06-08

Task:

```text
KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
```

Checkpoints:

```text
Phase ACT:
checkpoints/libero_long_corrected_task2/act_chunked_corrected_h20_task2_phase_conditioned/best.pt

Event-gated ACT:
checkpoints/libero_long_corrected_task2/event_gated_act_h20_task2_phase_memory/best.pt
```

Training:

```text
Phase ACT stopped at epoch 30.
Best phase checkpoint came from epoch 29 with val_loss=0.034255.

Event-gated ACT stopped after epoch 50.
Epoch 50 was the best event checkpoint with val_loss=0.022431.
```

Offline eval:

| Model | continuous_mse | continuous_mae | gripper_sign_accuracy |
|---|---:|---:|---:|
| Phase ACT | 0.033583 | 0.132092 | 0.984755 |
| Event-gated ACT | 0.022641 | 0.107986 | 0.989935 |

Offline deltas:

```text
continuous_mse improved by about 32.6%
continuous_mae improved by about 18.3%
gripper sign accuracy improved by about 0.52 percentage points
```

Rollout results:

| Split | Phase ACT | Event-gated ACT |
|---|---:|---:|
| train10 | 9/10 | 10/10 |
| val5 | 2/5 | 5/5 |
| test5 | 4/5 | 4/5 |
| total | 15/20 | 19/20 |
| held-out val+test | 6/10 | 9/10 |

Test split details:

| episode_idx | Phase ACT | Event-gated ACT |
|---:|---:|---:|
| 2 | success | success |
| 7 | failure | success |
| 11 | success | success |
| 20 | success | failure |
| 22 | success | success |

Interpretation:

```text
Task 2 is another positive event-memory result. Event-gated ACT clearly improves offline
continuous action prediction and improves rollout success from 15/20 to 19/20 under the
same selected episode protocol.

The strongest rollout gain is on validation, where event-gated improves from 2/5 to 5/5.
The test split remains tied at 4/5, but the failure moved from episode 7 to episode 20.
That means event memory did not simply dominate every case, but it substantially improved
overall robustness on this 20-episode protocol.
```

Detailed rollout analysis:

```text
Phase ACT failures:
- train episode 6
- val episodes 29, 40, 41
- test episode 7

Event-gated ACT failures:
- test episode 20
```

Flipped cases:

| Split | episode_idx | Phase ACT | Event-gated ACT |
|---|---:|---:|---:|
| train | 6 | failure at 300 steps | success at 217 steps |
| val | 29 | failure at 300 steps | success at 246 steps |
| val | 40 | failure at 300 steps | success at 257 steps |
| val | 41 | failure at 300 steps | success at 283 steps |
| test | 7 | failure at 300 steps | success at 253 steps |
| test | 20 | success at 245 steps | failure at 300 steps |

Success-speed summary:

| Split | Phase ACT avg success steps | Event-gated ACT avg success steps |
|---|---:|---:|
| train | 261.7 | 237.5 |
| val | 245.0 | 257.6 |
| test | 248.0 | 242.0 |

Trace findings:

```text
The flipped task-2 failures are not early phase-recognition failures. Both policies progress
through current_phase_id 0 -> 1 -> 2 at the same scheduled phase boundaries in the traced
episodes. The failures happen after reaching phase 2, so the likely issue is late-stage
trajectory quality / placement control rather than phase conditioning itself.

The gripper classifier is also not the main weakness. Offline gripper sign accuracy improves
slightly for event-gated ACT, and rollout traces show both policies execute open/close
transitions. The difference is more about whether the late motion successfully completes
the stove/moka-pot placement condition.
```

Swapped test episodes:

```text
Episode 7:
- Phase ACT failed at 300 steps.
- Event-gated ACT succeeded at 253 steps.
- Both entered phase 2 at step 160.
- Event-gated used fewer gripper toggles and maintained stronger late motion, which is
  consistent with a cleaner placement/execution sequence.

Episode 20:
- Phase ACT succeeded at 245 steps.
- Event-gated ACT failed at 300 steps.
- Both entered phase 2 at step 160.
- Event-gated showed extra late gripper toggles and lower late end-effector/action deltas,
  suggesting it got into a weak or stalled late-stage behavior despite good offline metrics.
```

Decision:

```text
This is a real positive result for the per-task event-memory protocol, especially combined
with task 5. Task 2 supports continuing the same protocol on the next task rather than
pivoting architecture immediately.

The caveat is that the test split is tied at 4/5, so the strongest claim should be:
event-gated improves task-2 offline action prediction and improves the 20-episode rollout
sample from 15/20 to 19/20, while held-out test-only remains tied on this small sample.
```

Artifacts:

```text
results/baselines_corrected_task2.csv
results/libero_rollouts_phase_act_task2_train10.csv
results/libero_rollouts_phase_act_task2_val5.csv
results/libero_rollouts_phase_act_task2_test5.csv
results/libero_rollouts_event_gated_act_task2_train10.csv
results/libero_rollouts_event_gated_act_task2_val5.csv
results/libero_rollouts_event_gated_act_task2_test5.csv
results/rollout_trace_phase_act_task2_train10.csv
results/rollout_trace_phase_act_task2_val5.csv
results/rollout_trace_phase_act_task2_test5.csv
results/rollout_trace_event_gated_act_task2_train10.csv
results/rollout_trace_event_gated_act_task2_val5.csv
results/rollout_trace_event_gated_act_task2_test5.csv
```

Diagnostic videos:

```text
Generated after the original measured rollouts, using a small diagnostic split:
splits/libero_long_task2_video_diagnostics.txt

Episodes:
6, 7, 20, 29, 40, 41

Phase ACT videos:
results/rollout_videos_phase_act_task2_diagnostics/act_chunked_corrected_h20_task2_phase_conditioned/

Event-gated ACT videos:
results/rollout_videos_event_gated_act_task2_diagnostics/event_gated_act_h20_task2_phase_memory/

Video rerun CSVs:
results/libero_rollouts_phase_act_task2_video_diagnostics.csv
results/libero_rollouts_event_gated_act_task2_video_diagnostics.csv

Video rerun traces:
results/rollout_trace_phase_act_task2_video_diagnostics.csv
results/rollout_trace_event_gated_act_task2_video_diagnostics.csv
```

Video-rerun caveat:

```text
The diagnostic video rerun should be used for qualitative inspection, not as a replacement
for the original measured table above. The rerun reproduced the main phase failures on
episodes 6, 7, and 29, but phase episodes 40 and 41 succeeded on video rerun despite
failing in the original val5 run. Event-gated episode 20 also succeeded on video rerun
despite failing in the original test5 run.

This indicates some rollout nondeterminism or context sensitivity in the simulator/policy
execution path. The original 20-episode table remains the primary measured result, while
the videos are diagnostic artifacts for inspecting behavior.
```
