# Diverse Open X Subset

This builds an additional non-Franka Open X subset without touching the existing Franka data paths.

Default mix:

- `berkeley_autolab_ur5`: UR5
- `roboturk`: Sawyer
- `stanford_kuka_multimodal_dataset_converted_externally_to_rlds`: Kuka
- `bridge`: WidowX

Output paths:

- raw: `data/raw_diverse`
- processed: `data/processed_diverse`
- episode shards: `data/episode_shards_diverse_5k`

## Build Locally

```bash
uv sync --extra openx

uv run python scripts/build_diverse_openx_subset.py \
  --overwrite
```

To change the mix:

```bash
uv run python scripts/build_diverse_openx_subset.py \
  --mix berkeley_autolab_ur5=900 roboturk=1600 stanford_kuka_multimodal_dataset_converted_externally_to_rlds=1250 bridge=1250 \
  --overwrite
```

## Upload Shards

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf upload-large-folder \
  Alcatraz1412/vla-franka-subset \
  data/episode_shards_diverse_5k \
  --repo-type dataset \
  --include "shards/*.tar" \
  --include "splits/*.json" \
  --include "manifest.jsonl" \
  --num-workers 1
```

This uploads only tar shards and split metadata. It does not overwrite existing Franka shards unless remote paths are identical.
