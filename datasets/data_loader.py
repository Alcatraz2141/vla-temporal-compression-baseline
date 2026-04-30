from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from datasets.streaming_vla_dataset import build_streaming_dataset
from datasets.vla_dataset import VLADataset, seed_worker, vla_collate_fn


def build_dataloader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    data_cfg = config["data"]
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    model_cfg = config.get("model", {})
    batch_size = train_cfg.get("batch_size", 32) if split == data_cfg.get("split", "train") else eval_cfg.get("batch_size", 32)
    augment_cfg = data_cfg.get("augment", {})
    source = data_cfg.get("source", "local")
    if source == "webdataset":
        urls_cfg = data_cfg.get("webdataset", {})
        urls = urls_cfg.get(f"{split}_urls") or urls_cfg.get("urls")
        if not urls:
            root = Path(urls_cfg.get("root", "data/webdataset"))
            urls = str(root / split / "shard-{000000..999999}.tar")
        return build_streaming_dataset(
            urls=urls,
            image_size=int(data_cfg.get("image_size", 224)),
            batch_size=int(batch_size),
            shuffle=shuffle,
            augment=bool(augment_cfg.get("enabled", False)) and split == data_cfg.get("split", "train"),
            stats_path=data_cfg.get("normalization", {}).get("stats_path") or urls_cfg.get("stats_path"),
        )

    T_obs = int(data_cfg["T_obs"])
    if model_cfg.get("baseline", "sliding_window") == "larger_window":
        T_obs *= 2
    dataset = VLADataset(
        root=Path(data_cfg.get("processed_root", data_cfg["root"])),
        split=split,
        T_obs=T_obs,
        T_action=data_cfg["T_action"],
        image_size=data_cfg.get("image_size", 224),
        baseline=model_cfg.get("baseline", "sliding_window"),
        augment=bool(augment_cfg.get("enabled", False)) and split == data_cfg.get("split", "train"),
        stats_path=data_cfg.get("normalization", {}).get("stats_path"),
    )

    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 42)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=vla_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )
