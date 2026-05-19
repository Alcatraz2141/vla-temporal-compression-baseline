from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.data_loader import build_dataloader
from models.vla_baseline import build_model
from utils.config import load_config
from utils.metrics import masked_mse, temporal_smoothness
from utils.seed import resolve_device, set_seed


def amp_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {name}")


def autocast_context(device: torch.device, cfg: dict[str, Any]) -> torch.amp.autocast_mode.autocast:
    enabled = bool(cfg["training"].get("amp", False)) and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=amp_dtype(cfg["training"].get("amp_dtype", "bfloat16")), enabled=enabled)


def _safe_len(loader: torch.utils.data.DataLoader) -> int:
    try:
        return len(loader)
    except TypeError:
        return 1


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def infer_dims(loader: torch.utils.data.DataLoader, config: dict[str, Any]) -> tuple[int, int]:
    if config["data"].get("source") == "webdataset":
        if config["data"].get("state_dim") is None or config["data"].get("action_dim") is None:
            raise ValueError("Streaming WebDataset training requires data.state_dim and data.action_dim in the config.")
        return int(config["data"]["state_dim"]), int(config["data"]["action_dim"])
    sample = loader.dataset[0]
    if "recent_states" in sample:
        state_dim = int(config["data"].get("state_dim") or sample["recent_states"].shape[-1])
        action_dim = int(config["data"].get("action_dim") or sample["target_actions"].shape[-1])
        return state_dim, action_dim
    state_dim = int(config["data"].get("state_dim") or sample["states"].shape[-1])
    action_dim = int(config["data"].get("action_dim") or sample["actions"].shape[-1])
    return state_dim, action_dim


def _move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _model_forward_and_target(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = _move_tensor_batch(batch, device)
    if "recent_obs" in batch:
        if hasattr(model, "_orig_mod"):
            baseline = getattr(model._orig_mod, "__class__").__name__
        else:
            baseline = getattr(model, "__class__").__name__
        if baseline == "EventGatedMemoryVLA":
            pred = model(**batch)
        else:
            pred = model(images=batch["recent_obs"], states=batch["recent_states"])
        return pred, batch["target_actions"], batch["target_mask"]
    pred = model(images=batch["images"], states=batch["states"])
    return pred, batch["actions"], batch["mask"]


def _move_batch(batch: dict[str, Any], device: torch.device, channels_last: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["images"].to(device, non_blocking=True)
    states = batch["states"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    return images, states, actions, mask


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: dict[str, Any],
) -> float:
    model.train()
    total = 0.0
    batches = 0
    smooth_weight = float(cfg["training"].get("temporal_smoothness_weight", 0.0))
    max_steps = cfg["training"].get("max_steps_per_epoch")
    channels_last = bool(cfg["training"].get("channels_last", False)) and device.type == "cuda"
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        if max_steps is not None and step > int(max_steps):
            break
        with autocast_context(device, cfg):
            pred, actions, mask = _model_forward_and_target(model, batch, device)
            loss = masked_mse(pred, actions, mask)
            if smooth_weight > 0:
                loss = loss + smooth_weight * temporal_smoothness(pred)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip_norm", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        batches += 1

        if step % int(cfg["training"].get("log_every", 20)) == 0:
            print(f"step={step} loss={total / step:.6f}")
    return total / max(batches or _safe_len(loader), 1)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    batches = 0
    channels_last = False
    for batch in tqdm(loader, desc="val", leave=False):
        pred, actions, mask = _model_forward_and_target(model, batch, device)
        loss = masked_mse(pred, actions, mask)
        total += float(loss.cpu())
        batches += 1
    return total / max(batches or _safe_len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PyTorch VLA baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
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

    train_loader = build_dataloader(cfg, cfg["data"].get("split", "train"), shuffle=True)
    val_loader = build_dataloader(cfg, cfg["data"].get("val_split", "val"), shuffle=False)
    state_dim, action_dim = infer_dims(train_loader, cfg)
    model = build_model(cfg, state_dim, action_dim).to(device)
    if bool(cfg["training"].get("channels_last", False)) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["training"].get("amp", False)) and device.type == "cuda")

    run_name = cfg["model"].get("run_name", cfg["model"].get("baseline", "sliding_window"))
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints")) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    start_epoch = 1
    resume = cfg["training"].get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val", checkpoint.get("val_mse", best_val)))
    if bool(cfg["training"].get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model, mode=cfg["training"].get("compile_mode", "reduce-overhead"))

    for epoch in range(start_epoch, int(cfg["training"].get("epochs", 5)) + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, cfg)
        val_loss = validate(model, val_loader, device)
        print(f"epoch={epoch} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")
        checkpoint = {
            "model": _unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "epoch": epoch,
            "val_mse": val_loss,
            "best_val": min(best_val, val_loss),
        }
        torch.save(checkpoint, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
