# Task 2 Final Control Audit

Date: 2026-06-08

Task:

```text
KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
```

Question:

```text
Is the task-2 event-gated result real, or could it be explained by extra training,
extra memory parameters, age/recency memory, or a validation-split mistake?
```

Models compared:

| Model | Checkpoint | Epoch |
|---|---|---:|
| Phase ACT original | `checkpoints/libero_long_corrected_task2/act_chunked_corrected_h20_task2_phase_conditioned/best.pt` | 29 |
| Phase ACT continued20 | `checkpoints/libero_long_corrected_task2/act_chunked_corrected_h20_task2_phase_continued20/best.pt` | 49 |
| Age-gated ACT | `checkpoints/libero_long_corrected_task2/age_gated_act_h20_task2_phase_memory20/last.pt` | 49 |
| Event-gated ACT | `checkpoints/libero_long_corrected_task2/event_gated_act_h20_task2_phase_memory/best.pt` | 50 |

Training-control status:

```text
Phase ACT continued20, age-gated ACT, and event-gated ACT all start from the same task-2
phase ACT checkpoint.

Phase ACT continued20 and age-gated ACT run the same 20-epoch continuation budget.
Event-gated ACT ran to epoch 50 from the same epoch-29 phase checkpoint, effectively the
same continuation scale.
```

Validation-split caveat:

```text
Some training checkpoints embed val_split=train because earlier monitoring used train-window
validation. That makes their training-time val_loss and the stock eval.py rows imperfect
for held-out claims.

This does not directly affect the online rollout CSVs, because rollouts use explicit LIBERO
split files and simulator init states.
```

Corrected held-out offline eval:

```text
file: results/task2_fixed_split_offline_eval_20260608.csv
splits: val, test
eval_windows_per_episode: 4
task filter: task 2 only
```

Held-out offline continuous MSE:

| Model | val continuous_mse | test continuous_mse |
|---|---:|---:|
| Phase ACT original | 0.326906 | 0.251335 |
| Phase ACT continued20 | 0.325022 | 0.233839 |
| Age-gated ACT | 0.329651 | 0.253350 |
| Event-gated ACT | 0.315877 | 0.250441 |

Held-out offline continuous MAE:

| Model | val continuous_mae | test continuous_mae |
|---|---:|---:|
| Phase ACT original | 0.375442 | 0.344403 |
| Phase ACT continued20 | 0.374775 | 0.330204 |
| Age-gated ACT | 0.371907 | 0.350297 |
| Event-gated ACT | 0.376062 | 0.342903 |

Interpretation of held-out offline eval:

```text
Held-out offline action prediction does not explain the rollout result.
Event-gated is slightly best on val MSE, but phase-continued is best on test MSE/MAE.
Age-gated is not obviously catastrophic offline, yet it collapses in rollout.

Therefore offline action prediction is not sufficient for judging these policies.
The main evidence is closed-loop rollout.
```

Closed-loop rollout result:

| Model | train10 | val5 | test5 | total | held-out val+test |
|---|---:|---:|---:|---:|---:|
| Phase ACT original | 9/10 | 2/5 | 4/5 | 15/20 | 6/10 |
| Phase ACT continued20 | 6/10 | 3/5 | 3/5 | 12/20 | 6/10 |
| Age-gated ACT | 3/10 | 2/5 | 0/5 | 5/20 | 2/10 |
| Event-gated ACT | 10/10 | 5/5 | 4/5 | 19/20 | 9/10 |

Episode-level rollout summary:

```text
Event-gated ACT succeeds on every selected train and val episode.
Event-gated ACT test failure: episode 20.

Age-gated ACT fails every selected test episode:
  2, 7, 11, 20, 22

Phase continued20 fixes some original phase failures but breaks several original successes:
  new train failures: 0, 4, 10, 12
  new test failures: 11, 22
```

Final audit conclusion:

```text
The task-2 event-gated result is not explained by longer training alone:
  phase continued20 gets 12/20 while event-gated gets 19/20.

It is not explained by simply adding older-context memory with an age/recency prior:
  age-gated gets 5/20 while event-gated gets 19/20.

It is not explained by the training-time validation split:
  corrected held-out offline eval does not favor event-gated enough to explain the rollout gap,
  and rollout selection uses explicit split files independent of eval.py.

The clean claim is:
  event-gated ACT gives substantially better task-2 closed-loop rollout robustness than
  phase ACT, equal-budget continued phase ACT, and age-gated memory ACT under this selected
  split-aware 20-episode protocol.
```

Remaining gaps before paper-grade claim:

```text
1. Single seed only.
2. Small selected rollout set: train10 / val5 / test5.
3. Some rollout nondeterminism was observed in diagnostic video reruns.
4. Event-gated test-only is tied with original phase ACT at 4/5.
5. Broader tasks and/or larger confirmation rollouts are still needed.
```

Recommended next step:

```text
Run either:
  A. same control protocol on another task, or
  B. larger task-2 confirmation rollout with fixed checkpoint choices:
     phase original, phase continued20, age-gated, event-gated.

Do not add another architecture change until this control story is preserved.
```
