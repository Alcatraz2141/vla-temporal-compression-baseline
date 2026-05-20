from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.data_loader import build_dataloader
from models.vla_baseline import build_model
from utils.config import load_config
from utils.language import language_ids
from utils.metrics import masked_mse, temporal_smoothness
from utils.seed import resolve_device, set_seed


def _move_tensor_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _model_forward_and_target(
    model: torch.nn.Module,
    batch: dict[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = _move_tensor_batch(batch, device)
    if "recent_obs" in batch:
        model_name = model.__class__.__name__
        if model_name == "EventGatedMemoryVLA":
            if bool(getattr(model, "use_language", False)) and "language" in batch:
                vocab_size = int(getattr(model, "language_embedding").num_embeddings)
                batch["language_ids"] = language_ids(batch["language"], vocab_size, device)
            pred = model(**batch)
        else:
            kwargs = {}
            if bool(getattr(model, "use_language", False)) and "language" in batch:
                vocab_size = int(getattr(model, "language_embedding").num_embeddings)
                kwargs["language_ids"] = language_ids(batch["language"], vocab_size, device)
            pred = model(
                images=batch["recent_obs"],
                states=batch["recent_states"],
                actions=batch["recent_actions"],
                **kwargs,
            )
        return pred, batch["target_actions"], batch["target_mask"]
    pred = model(images=batch["images"], states=batch["states"])
    return pred, batch["actions"], batch["mask"]


def _safe_len(loader: torch.utils.data.DataLoader) -> int:
    try:
        return len(loader)
    except TypeError:
        return 1


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_smoothness = 0.0
    batches = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        pred, actions, mask = _model_forward_and_target(model, batch, device)
        mse = masked_mse(pred, actions, mask)
        mae = (pred - actions).abs()
        mae = (mae * mask.to(mae.dtype).unsqueeze(-1)).sum() / mask.to(mae.dtype).sum().clamp_min(1.0)
        total_mse += float(mse.cpu())
        total_mae += float(mae.cpu())
        total_smoothness += float(temporal_smoothness(pred).cpu())
        batches += 1
    mse = total_mse / max(batches or _safe_len(loader), 1)
    denom = max(batches or _safe_len(loader), 1)
    return {
        "mse": mse,
        "mae": total_mae / denom,
        "pred_temporal_smoothness": total_smoothness / denom,
        "success_rate": float("nan"),
        "rollout_consistency": float("nan"),
    }


def append_results(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    mode = "a"
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        if header and header != list(row.keys()):
            existing = path.read_text(encoding="utf-8")
            backup = path.with_suffix(path.suffix + ".bak")
            backup.write_text(existing, encoding="utf-8")
            exists = False
            mode = "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained VLA baseline checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--baseline", choices=["sliding_window", "no_temporal", "larger_window", "bc_resnet50", "rt1_style", "octo", "event_gated_memory"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.baseline is not None:
        cfg["model"]["baseline"] = args.baseline
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.checkpoint_dir is not None:
        cfg["training"]["checkpoint_dir"] = str(args.checkpoint_dir)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    run_name = cfg["model"].get("run_name", cfg["model"].get("baseline", "sliding_window"))
    default_variant_checkpoint = Path(cfg["training"].get("checkpoint_dir", "checkpoints")) / run_name / "best.pt"
    checkpoint_path = args.checkpoint or (default_variant_checkpoint if default_variant_checkpoint.exists() else Path(cfg["evaluation"].get("checkpoint", "checkpoints/best.pt")))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_cfg = checkpoint.get("config", cfg)
    ckpt_cfg["model"]["baseline"] = cfg["model"].get("baseline", ckpt_cfg["model"].get("baseline", "sliding_window"))
    ckpt_cfg["model"]["run_name"] = cfg["model"].get("run_name", ckpt_cfg["model"].get("run_name", ckpt_cfg["model"]["baseline"]))

    loader = build_dataloader(ckpt_cfg, ckpt_cfg["data"].get("val_split", "val"), shuffle=False)
    model = build_model(ckpt_cfg, int(checkpoint["state_dim"]), int(checkpoint["action_dim"])).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate(model, loader, device)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline": ckpt_cfg["model"].get("run_name", ckpt_cfg["model"].get("baseline", "sliding_window")),
        "checkpoint": str(checkpoint_path),
        "seed": ckpt_cfg.get("seed", 42),
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "pred_temporal_smoothness": metrics["pred_temporal_smoothness"],
        "success_rate": metrics["success_rate"],
        "rollout_consistency": metrics["rollout_consistency"],
    }
    append_results(Path(ckpt_cfg["evaluation"].get("results_path", "results/baselines.csv")), row)
    print(row)


if __name__ == "__main__":
    main()
