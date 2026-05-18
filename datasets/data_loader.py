from __future__ import annotations

from pathlib import Path
from typing import Any

from braceexpand import braceexpand
import torch
from torch.utils.data import DataLoader

from datasets.episode_dataset import EpisodeWindowDataset, episode_collate_fn
from datasets.episode_shard_dataset import EpisodeShardWindowDataset
from datasets.libero_long_dataset import LiberoLongDataset
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
    if source in {"episode", "episode_shards"}:
        episode_cfg = data_cfg.get("episode", {})
        memory_cfg = config.get("memory", {})
        dataset_cls = EpisodeShardWindowDataset if source == "episode_shards" else EpisodeWindowDataset
        dataset = dataset_cls(
            root=Path(episode_cfg.get("root", "data/episodes")),
            split=split,
            K_recent=int(data_cfg.get("K_recent", data_cfg.get("T_obs", 4))),
            H_action=int(data_cfg.get("H_action", data_cfg.get("T_action", 16))),
            image_size=int(data_cfg.get("image_size", 224)),
            max_older_steps=int(episode_cfg.get("max_older_steps", int(memory_cfg.get("chunk_size", 8)) * int(memory_cfg.get("max_memory_tokens", 8)))),
            seed=int(config.get("seed", 42)),
            augment=bool(augment_cfg.get("enabled", False)) and split == data_cfg.get("split", "train"),
            stats_path=data_cfg.get("normalization", {}).get("stats_path"),
            samples_per_epoch=episode_cfg.get("samples_per_epoch") if split == data_cfg.get("split", "train") else None,
            eval_windows_per_episode=int(episode_cfg.get("eval_windows_per_episode", 1)),
        )
        generator = torch.Generator()
        generator.manual_seed(int(config.get("seed", 42)))
        num_workers = int(data_cfg.get("num_workers", 2))
        kwargs: dict[str, Any] = {
            "batch_size": int(batch_size),
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0,
            "collate_fn": episode_collate_fn,
            "worker_init_fn": seed_worker,
            "generator": generator,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
        return DataLoader(dataset, **kwargs)
    if source == "libero_long":
        libero_cfg = data_cfg.get("libero_long", {})
        memory_cfg = config.get("memory", {})
        dataset = LiberoLongDataset(
            root=Path(libero_cfg.get("root", "data/libero_long")),
            split=split,
            K_recent=int(data_cfg.get("K_recent", data_cfg.get("T_obs", 8))),
            H_action=int(data_cfg.get("H_action", data_cfg.get("T_action", 4))),
            image_size=int(data_cfg.get("image_size", 224)),
            max_older_steps=int(libero_cfg.get("max_older_steps", int(memory_cfg.get("chunk_size", 4)) * int(memory_cfg.get("max_memory_tokens", 16)))),
            seed=int(config.get("seed", 42)),
            augment=bool(augment_cfg.get("enabled", False)) and split == data_cfg.get("split", "train"),
            hdf5_glob=str(libero_cfg.get("hdf5_glob", "**/*.hdf5")),
            split_dir=libero_cfg.get("split_dir"),
            samples_per_epoch=libero_cfg.get("samples_per_epoch") if split == data_cfg.get("split", "train") else None,
            eval_windows_per_episode=int(libero_cfg.get("eval_windows_per_episode", 1)),
            max_episodes=libero_cfg.get("max_episodes"),
        )
        generator = torch.Generator()
        generator.manual_seed(int(config.get("seed", 42)))
        num_workers = int(data_cfg.get("num_workers", 2))
        kwargs = {
            "batch_size": int(batch_size),
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0,
            "collate_fn": episode_collate_fn,
            "worker_init_fn": seed_worker,
            "generator": generator,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
        return DataLoader(dataset, **kwargs)
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
        num_workers = int(data_cfg.get("num_workers", 4))
        if isinstance(urls, list):
            capped_workers = min(num_workers, max(len(urls), 1))
            if capped_workers != num_workers:
                print(
                    f"Capping WebDataset workers from {num_workers} to {capped_workers} for {len(urls)} local shard(s).",
                    flush=True,
                )
            num_workers = capped_workers
        return build_streaming_dataset(
            urls=urls,
            image_size=int(data_cfg.get("image_size", 224)),
            T_obs=T_obs,
            T_action=int(data_cfg["T_action"]),
            batch_size=int(batch_size),
            shuffle=shuffle,
            augment=bool(augment_cfg.get("enabled", False)) and split == data_cfg.get("split", "train"),
            stats_path=data_cfg.get("normalization", {}).get("stats_path") or urls_cfg.get("stats_path"),
            num_workers=num_workers,
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
