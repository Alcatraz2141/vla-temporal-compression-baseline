# Task-2 Age-Gated ACT Seed 187 Epoch-27 Stop

Date: 2026-07-01

```text
config: configs/paper_age_gated_act_task2_seed187.yaml
resume config: configs/paper_age_gated_act_task2_seed187_resume.yaml
checkpoint: checkpoints/paper_age_gated_task2_seed187/age_gated_act_h20_task2_phase_memory_seed187/last.pt
stopped after completed epoch: 27
best checkpoint by training validation: epoch 24
best val_mse: 0.05992962270975113
log: logs/paper_age_gated_task2_seed187_20260701.log
```

Training state:

```text
epoch 24: train_loss 0.045836, val_loss 0.059930
epoch 25: train_loss 0.046007, val_loss 0.073385
epoch 26: train_loss 0.043826, val_loss 0.066062
epoch 27: train_loss 0.043381, val_loss 0.061624
```

The stop request arrived just after epoch 28 had started; the log contains an epoch-28
`step=50` line. The saved checkpoint remains epoch 27, and partial epoch-28 work should be
treated as intentionally discarded.

Artifact backup:

```text
local: /workspace/run_backups/vla_run_artifacts_20260701_182414.tar.gz
Hugging Face commit: https://huggingface.co/datasets/Alcatraz1412/vla-run-backups/commit/dfc6e281436abc176783f2f59c57b4ce87e6d95d
```

Next step:

```bash
uv run python train.py --config configs/paper_age_gated_act_task2_seed187_resume.yaml
```

Resume from epoch 27 `last.pt` if continuing this run. Run offline eval and split-aware
train30/val5/test5 rollouts only after the selected reporting checkpoint is chosen.
