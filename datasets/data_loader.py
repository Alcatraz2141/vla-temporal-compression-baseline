from __future__ import annotations

from pathlib import Path
from typing import Any

from braceexpand import braceexpand
import torch
from torch.utils.data import DataLoader

from datasets.streaming_vla_dataset import build_streaming_dataset
from datasets.vla_dataset import VLADataset, seed_worker, vla_collate_fn


def _is_remote_url(url: str) -> bool:
    return "://" in url or url.startswith("pipe:") or url.startswith("hf://")


def _expand_local_shards(urls: str | list[str]) -> str | list[str]:
    if isinstance(urls, list):
        expanded = []
        for url in urls:
            item = _expand_local_shards(url)
            expanded.extend(item if isinstance(item, list) else [item])
        return expanded
    if _is_remote_url(urls):
        return urls
    candidates = list(braceexpand(urls)) if "{" in urls and "}" in urls else [urls]
    if len(candidates) == 1 and any(ch in candidates[0] for ch in "*?[]"):
        candidates = [str(path) for path in sorted(Path().glob(candidates[0]))]
    existing = [path for path in candidates if Path(path).exists()]
    if not existing:
        raise FileNotFoundError(
            f"No WebDataset shards matched {urls!r}. Check data.webdataset URLs or download/export shards first."
        )
    missing = len(candidates) - len(existing)
    if missing > 0:
        print(f"Using {len(existing)} existing shard(s); skipped {missing} missing shard path(s) from {urls!r}.", flush=True)
    return existing


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
        urls = _expand_local_shards(urls)
        T_obs = int(data_cfg["T_obs"])
        if model_cfg.get("baseline", "sliding_window") == "no_temporal":
            T_obs = 1
        elif model_cfg.get("baseline", "sliding_window") == "larger_window":
            T_obs *= 2
        return build_streaming_dataset(
            urls=urls,
            image_size=int(data_cfg.get("image_size", 224)),
            T_obs=T_obs,
            T_action=int(data_cfg["T_action"]),
            batch_size=int(batch_size),
            shuffle=shuffle,
            augment=bool(augment_cfg.get("enabled", False)) and split == data_cfg.get("split", "train"),
            stats_path=data_cfg.get("normalization", {}).get("stats_path") or urls_cfg.get("stats_path"),
            num_workers=int(data_cfg.get("num_workers", 4)),
            prefetch_factor=int(data_cfg.get("prefetch_factor", 4)),
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
    num_workers = int(data_cfg.get("num_workers", 2))
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "collate_fn": vla_collate_fn,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
    return DataLoader(dataset, **kwargs)
