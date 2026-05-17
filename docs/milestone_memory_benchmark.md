# Milestone: Episode-Level Long-Horizon Memory Benchmark

This milestone keeps the existing local/WebDataset download and Hugging Face upload path intact, but adds an episode-level data path for long-horizon memory experiments.

## Why Episode-Level Data

The previous WebDataset export stores overlapping training windows. That is fine for short-context behavior cloning, but it repeats images and can leak near-identical windows across splits if split generation is careless. The episode format stores each trajectory once, then constructs windows inside the dataloader.

Each exported episode contains:

- `images/`
- `actions.npy`
- `states.npy`
- `metadata.json`

Split files live in `splits/train.json`, `splits/val.json`, and `splits/test.json`, so windows from one episode never cross into another split.

## Create A Small Episode Dataset

```bash
uv run python scripts/export_episode_dataset.py \
  --input-root data/processed \
  --output-root data/episodes_smoke \
  --max-episodes 50 \
  --overwrite
```

## Run Sliding Window

```bash
uv run python training/train.py --config configs/milestone_episode_sliding.yaml
uv run python evaluation/eval.py --config configs/milestone_episode_sliding.yaml
```

## Run Event-Gated Memory

```bash
uv run python training/train.py --config configs/milestone_event_memory.yaml
uv run python evaluation/eval.py --config configs/milestone_event_memory.yaml
```

Both evaluations append to `results/baselines.csv`.

## Current Baselines

- `sliding_window`: short recent context only. It ignores `older_*` fields.
- `event_gated_memory`: keeps recent tokens at full fidelity, chunks older context, compresses each chunk with lightweight attention pooling, gates chunk summaries using visual/action/state deltas plus age, then predicts the action chunk.

## Next After Smoke

Scale the same configs to a larger `data/episodes` root on RunPod, then add the RB-VLA-style recursive belief baseline and the first gate/query ablations.
