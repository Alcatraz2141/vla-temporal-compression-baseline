from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return normalize_config(cfg)


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Accept both legacy nested configs and compact milestone configs."""
    if not isinstance(cfg.get("model"), str):
        return cfg

    dataset = cfg.get("dataset", "fractal")
    model_name = cfg.get("model", "sliding_window")
    batch_size = int(cfg.get("batch_size", 32))
    total_steps = int(cfg.get("total_steps", 50_000))
    root_by_dataset = {
        "libero_long": "data/libero_long",
        "fractal": "data/raw",
        "ur5": "data/raw_diverse",
        "others": "data/raw_diverse",
    }
    return {
        "dataset": dataset,
        "seed": int(cfg.get("seed", 42)),
        "device": cfg.get("device", "auto"),
        "K_recent": int(cfg.get("K_recent", 8)),
        "H_action": int(cfg.get("H_action", 4)),
        "chunk_size": int(cfg.get("chunk_size", 4)),
        "max_memory_tokens": int(cfg.get("max_memory_tokens", 16)),
        "gate_type": cfg.get("gate_type", "none"),
        "query_type": cfg.get("query_type", "none"),
        "data": {
            "source": "unified_episode",
            "dataset": dataset,
            "split": "train",
            "val_split": "val",
            "image_size": int(cfg.get("image_size", 128)),
            "T_obs": int(cfg.get("K_recent", 8)),
            "T_action": int(cfg.get("H_action", 4)),
            "K_recent": int(cfg.get("K_recent", 8)),
            "H_action": int(cfg.get("H_action", 4)),
            "num_workers": int(cfg.get("num_workers", 2)),
            "episode_loader": {
                "root": cfg.get("data_root", root_by_dataset.get(dataset, "data/raw_diverse")),
                "split_dir": cfg.get("split_dir", "splits"),
                "hdf5_glob": cfg.get("hdf5_glob", "**/*.hdf5"),
                "samples_per_epoch": cfg.get("samples_per_epoch"),
                "eval_windows_per_episode": int(cfg.get("eval_windows_per_episode", 1)),
            },
            "augment": {"enabled": bool(cfg.get("augment", False))},
        },
        "model": {
            "baseline": model_name,
            "vision_encoder": "resnet18",
            "pretrained_vision": bool(cfg.get("pretrained_vision", True)),
            "d_model": int(cfg.get("d_model", 256)),
            "n_layers": int(cfg.get("n_layers", 4)),
            "n_heads": int(cfg.get("n_heads", 4)),
            "dropout": float(cfg.get("dropout", 0.1)),
            "state_hidden_dim": int(cfg.get("state_hidden_dim", 128)),
            "action_hidden_dim": int(cfg.get("action_hidden_dim", 256)),
        },
        "memory": {
            "chunk_size": int(cfg.get("chunk_size", 4)),
            "max_memory_tokens": int(cfg.get("max_memory_tokens", 16)),
            "gate_type": "event" if cfg.get("gate_type") == "none" else cfg.get("gate_type", "event"),
            "query_type": "concat" if cfg.get("query_type") == "none" else cfg.get("query_type", "cross_attention"),
        },
        "training": {
            "batch_size": batch_size,
            "epochs": int(cfg.get("epochs", 1)),
            "max_steps_per_epoch": total_steps,
            "lr": float(cfg.get("lr", 1.0e-4)),
            "weight_decay": float(cfg.get("weight_decay", 1.0e-4)),
            "grad_clip_norm": float(cfg.get("grad_clip_norm", 1.0)),
            "amp": bool(cfg.get("amp", True)),
            "amp_dtype": cfg.get("amp_dtype", "bfloat16"),
            "channels_last": bool(cfg.get("channels_last", True)),
            "compile": bool(cfg.get("compile", False)),
            "temporal_smoothness_weight": float(cfg.get("temporal_smoothness_weight", 0.0)),
            "log_every": int(cfg.get("log_every", 500)),
            "checkpoint_dir": cfg.get("checkpoint_dir", f"checkpoints/{dataset}"),
            "resume": cfg.get("resume"),
            "optimizer": cfg.get("optimizer", "adamw"),
            "lr_schedule": cfg.get("lr_schedule", "cosine"),
            "warmup_steps": int(cfg.get("warmup_steps", 1000)),
            "total_steps": total_steps,
            "eval_every": int(cfg.get("eval_every", 5000)),
            "save_every": int(cfg.get("save_every", 10000)),
        },
        "evaluation": {
            "batch_size": int(cfg.get("eval_batch_size", batch_size)),
            "results_path": cfg.get("results_path", "results/baselines.csv"),
        },
    }


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base
