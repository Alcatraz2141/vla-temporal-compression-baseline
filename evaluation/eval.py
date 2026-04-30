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
from utils.metrics import masked_mse
from utils.seed import resolve_device, set_seed


def _safe_len(loader: torch.utils.data.DataLoader) -> int:
    try:
        return len(loader)
    except TypeError:
        return 1


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_mse = 0.0
    batches = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        pred = model(images=batch["images"].to(device), states=batch["states"].to(device))
        mse = masked_mse(pred, batch["actions"].to(device), batch["mask"].to(device))
        total_mse += float(mse.cpu())
        batches += 1
    mse = total_mse / max(batches or _safe_len(loader), 1)
    return {"mse": mse, "success_rate": float("nan"), "rollout_consistency": float("nan")}


def append_results(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained VLA baseline checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--baseline", choices=["sliding_window", "no_temporal", "larger_window", "bc_resnet50", "octo"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.baseline is not None:
        cfg["model"]["baseline"] = args.baseline
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    default_variant_checkpoint = (
        Path(cfg["training"].get("checkpoint_dir", "checkpoints")) / cfg["model"].get("baseline", "sliding_window") / "best.pt"
    )
    checkpoint_path = args.checkpoint or (default_variant_checkpoint if default_variant_checkpoint.exists() else Path(cfg["evaluation"].get("checkpoint", "checkpoints/best.pt")))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_cfg = checkpoint.get("config", cfg)
    ckpt_cfg["model"]["baseline"] = cfg["model"].get("baseline", ckpt_cfg["model"].get("baseline", "sliding_window"))

    loader = build_dataloader(ckpt_cfg, ckpt_cfg["data"].get("val_split", "val"), shuffle=False)
    model = build_model(ckpt_cfg, int(checkpoint["state_dim"]), int(checkpoint["action_dim"])).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate(model, loader, device)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline": ckpt_cfg["model"].get("baseline", "sliding_window"),
        "checkpoint": str(checkpoint_path),
        "seed": ckpt_cfg.get("seed", 42),
        "mse": metrics["mse"],
        "success_rate": metrics["success_rate"],
        "rollout_consistency": metrics["rollout_consistency"],
    }
    append_results(Path(ckpt_cfg["evaluation"].get("results_path", "results/baselines.csv")), row)
    print(row)


if __name__ == "__main__":
    main()
