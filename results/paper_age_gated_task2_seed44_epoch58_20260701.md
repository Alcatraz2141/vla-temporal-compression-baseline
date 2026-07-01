# Task-2 Age-Gated ACT Seed 44 Epoch-58 Audit

Date: 2026-07-01

```text
config: configs/paper_age_gated_act_task2_seed44.yaml
resume config: configs/paper_age_gated_act_task2_seed44_resume.yaml
checkpoint: checkpoints/paper_age_gated_task2_seed44/age_gated_act_h20_task2_phase_memory_seed44/last.pt
stopped after completed epoch: 58
best checkpoint by training validation: epoch 56
best val_mse: 0.044679350405931476
```

Offline eval:

```text
epoch-58 last.pt continuous_mse: 0.05122858919203281
epoch-58 last.pt continuous_mae: 0.1402573436498642
epoch-58 last.pt gripper_sign_accuracy: 0.9779375076293946

best.pt continuous_mse: 0.03577113375067711
best.pt continuous_mae: 0.12979439795017242
best.pt gripper_sign_accuracy: 0.9845625162124634
```

Rollout on epoch-58 `last.pt`:

```text
train30 / val5 / test5 = 21/30, 3/5, 1/5 = 25/40
held-out val+test = 4/10
failure episode IDs:
  train: [4, 6, 8, 12, 17, 18, 19, 26, 35]
  val: [29, 40]
  test: [7, 11, 20, 22]
```

Artifact backup:

```text
local: /workspace/run_backups/vla_run_artifacts_20260701_142852.tar.gz
Hugging Face commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/cffb17c7e30e71e53d06ed368511abcf627601e2
```

Interpretation:

```text
Age-gated seed 44 reaches a strong train-split rollout count but weak held-out test rollout.
The validation-selected best.pt is substantially better offline than epoch-58 last.pt, so a later
checkpoint-selection audit should roll out best.pt before finalizing seed-44 reporting.

Compared with age-gated seed 43, epoch-58 seed 44 is similar on total rollout but weaker on
held-out:
  seed 43 epoch 60 last.pt: 26/40 total, 5/10 held-out
  seed 44 epoch 58 last.pt: 25/40 total, 4/10 held-out

This keeps age-gated memory competitive with event-gated ACT on task 2, but it does not resolve
the checkpoint-selection and online/offline mismatch. Do not claim event-gated ACT beats
age-gated ACT on task 2 without the seed-44 best.pt rollout audit and additional matched seeds.
```
