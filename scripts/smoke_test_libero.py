from __future__ import annotations

import argparse
import csv
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.data_loader import build_dataloader
from models.vla_baseline import build_model
from utils.metrics import masked_mse
from utils.seed import set_seed


def build_smoke_config(args: argparse.Namespace, baseline: str) -> dict[str, Any]:
    return {
        "dataset": "libero_long",
        "seed": args.seed,
        "device": args.device,
        "data": {
            "source": "libero_long",
            "libero_long": {
                "root": str(args.data_root),
                "hdf5_glob": args.hdf5_glob,
                "split_dir": str(args.data_root / "splits"),
                "samples_per_epoch": 5,
                "eval_windows_per_episode": 1,
                "max_older_steps": 64,
                "max_episodes": 5,
            },
            "split": "train",
            "val_split": "val",
            "image_size": args.image_size,
            "T_obs": 8,
            "T_action": 4,
            "K_recent": 8,
            "H_action": 4,
            "state_dim": None,
            "action_dim": None,
            "num_workers": 0,
            "augment": {"enabled": False},
        },
        "model": {
            "baseline": baseline,
            "vision_encoder": "resnet18",
            "pretrained_vision": False,
            "d_model": 64,
            "n_layers": 1,
            "n_heads": 4,
            "dropout": 0.1,
            "state_hidden_dim": 64,
            "action_hidden_dim": 128,
        },
        "memory": {
            "chunk_size": 4,
            "max_memory_tokens": 16,
            "gate_type": "event",
            "query_type": "cross_attention",
        },
        "training": {"batch_size": 5},
        "evaluation": {"batch_size": 5},
    }


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def run_one_model(cfg: dict[str, Any], batch: dict[str, Any], device: torch.device) -> float:
    state_dim = int(batch["recent_states"].shape[-1])
    action_dim = int(batch["target_actions"].shape[-1])
    model = build_model(cfg, state_dim, action_dim).to(device)
    model.eval()
    batch = move_tensor_batch(batch, device)
    if cfg["model"]["baseline"] == "event_gated_memory":
        pred = model(**batch)
    else:
        pred = model(images=batch["recent_obs"], states=batch["recent_states"])
    loss = masked_mse(pred, batch["target_actions"], batch["target_mask"])
    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite loss for {cfg['model']['baseline']}: {float(loss.detach().cpu())}")
    return float(loss.detach().cpu())


def append_smoke_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU smoke test for LIBERO-Long dataloader and milestone models.")
    parser.add_argument("--data-root", type=Path, default=Path("data/libero_long"))
    parser.add_argument("--hdf5-glob", default="**/*.hdf5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-path", type=Path, default=Path("results/smoke_test.csv"))
    args = parser.parse_args()

    try:
        set_seed(args.seed)
        device = torch.device(args.device)
        sliding_cfg = build_smoke_config(args, "sliding_window")
        loader = build_dataloader(sliding_cfg, "train", shuffle=False)
        batch = next(iter(loader))
        if len(batch["episode_id"]) > 5:
            raise ValueError("Smoke loader returned more than 5 episodes.")

        sliding_loss = run_one_model(sliding_cfg, batch, device)
        event_cfg = build_smoke_config(args, "event_gated_memory")
        event_loss = run_one_model(event_cfg, batch, device)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": "sliding_window+event_gated_memory",
            "loss": event_loss,
            "sliding_window_loss": sliding_loss,
            "event_gated_memory_loss": event_loss,
            "episodes_loaded": len(batch["episode_id"]),
        }
        append_smoke_result(args.results_path, row)
        print(row)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
