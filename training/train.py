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
from utils.language import language_ids
from utils.metrics import masked_mse, temporal_smoothness
from utils.seed import resolve_device, set_seed


def load_compatible_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load older checkpoints after small backward-compatible architecture edits."""
    model_state = model.state_dict()
    patched = dict(state_dict)
    gate_key = "gate.0.weight"
    if gate_key in patched and gate_key in model_state and patched[gate_key].shape != model_state[gate_key].shape:
        old_weight = patched[gate_key]
        new_weight = model_state[gate_key].clone()
        cols = min(old_weight.shape[1], new_weight.shape[1])
        new_weight[:, :cols] = old_weight[:, :cols]
        patched[gate_key] = new_weight
    model.load_state_dict(patched)


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


def action_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    gripper_loss_type = str(cfg["training"].get("gripper_loss_type", "mse"))
    gripper_weight = float(cfg["training"].get("gripper_loss_weight", 1.0))
    mask_f = mask.to(pred.dtype)

    if gripper_loss_type == "bce_sign" and pred.size(-1) >= 7:
        continuous_loss = (pred[..., :-1] - target[..., :-1]).pow(2)
        continuous_mask = mask_f.unsqueeze(-1)
        continuous_sum = (continuous_loss * continuous_mask).sum()
        continuous_count = (continuous_mask.expand_as(continuous_loss)).sum()

        gripper_target = (target[..., -1] > 0).to(pred.dtype)
        gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred[..., -1],
            gripper_target,
            reduction="none",
        )
        gripper_sum = (gripper_loss * mask_f).sum()
        gripper_count = mask_f.sum()
        numerator = continuous_sum + gripper_weight * gripper_sum
        denominator = continuous_count + gripper_weight * gripper_count
        return numerator / denominator.clamp_min(1.0)

    if gripper_loss_type != "mse":
        raise ValueError(f"Unsupported training.gripper_loss_type: {gripper_loss_type}")

    loss = (pred - target).pow(2)
    weights = torch.ones(pred.size(-1), dtype=loss.dtype, device=loss.device)
    if pred.size(-1) >= 7 and gripper_weight != 1.0:
        weights[-1] = gripper_weight
    loss = loss * weights.view(1, 1, -1)
    mask_expanded = mask_f.unsqueeze(-1)
    return (loss * mask_expanded).sum() / (mask_expanded * weights.view(1, 1, -1)).sum().clamp_min(1.0)


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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
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
            loss = action_loss(pred, actions, mask, cfg)
            if smooth_weight > 0:
                loss = loss + smooth_weight * temporal_smoothness(pred)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip_norm", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.detach().cpu())
        batches += 1

        if step % int(cfg["training"].get("log_every", 20)) == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"step={step} loss={total / step:.6f} lr={lr_now:.2e}")
    return total / max(batches or _safe_len(loader), 1)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, cfg: dict[str, Any]) -> float:
    model.eval()
    total = 0.0
    batches = 0
    channels_last = False
    for batch in tqdm(loader, desc="val", leave=False):
        pred, actions, mask = _model_forward_and_target(model, batch, device)
        loss = action_loss(pred, actions, mask, cfg)
        total += float(loss.cpu())
        batches += 1
    return total / max(batches or _safe_len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PyTorch VLA baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--baseline", choices=["sliding_window", "no_temporal", "larger_window", "bc_resnet50", "rt1_style", "octo", "event_gated_memory"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.baseline is not None:
        cfg["model"]["baseline"] = args.baseline
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.checkpoint_dir is not None:
        cfg["training"]["checkpoint_dir"] = str(args.checkpoint_dir)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.max_steps_per_epoch is not None:
        cfg["training"]["max_steps_per_epoch"] = args.max_steps_per_epoch
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

    scheduler = None
    lr_schedule = cfg["training"].get("lr_schedule", "constant")
    if lr_schedule == "cosine":
        steps_per_epoch = len(train_loader)
        max_steps_ep = cfg["training"].get("max_steps_per_epoch")
        if max_steps_ep is not None:
            steps_per_epoch = min(steps_per_epoch, int(max_steps_ep))
        total_steps = steps_per_epoch * int(cfg["training"].get("epochs", 5))
        warmup_steps = int(cfg["training"].get("warmup_steps", 0))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6),
            ],
            milestones=[warmup_steps],
        )

    run_name = cfg["model"].get("run_name", cfg["model"].get("baseline", "sliding_window"))
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints")) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    start_epoch = 1
    resume = cfg["training"].get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        load_compatible_state_dict(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val", checkpoint.get("val_mse", best_val)))
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
    if bool(cfg["training"].get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model, mode=cfg["training"].get("compile_mode", "reduce-overhead"))

    for epoch in range(start_epoch, int(cfg["training"].get("epochs", 5)) + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, cfg, scheduler=scheduler)
        val_loss = validate(model, val_loader, device, cfg)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
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
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()
        torch.save(checkpoint, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
