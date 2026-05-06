from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _require_lerobot() -> tuple[Any, Any, Any]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as exc:
        raise SystemExit(
            "LeRobot/SmolVLA is not installed. Run: uv sync --extra smolvla\n"
            "or: uv pip install 'lerobot[smolvla]'"
        ) from exc
    return LeRobotDataset, make_pre_post_processors, SmolVLAPolicy


def train(cfg: dict[str, Any], extra_args: list[str]) -> None:
    smol = cfg["smolvla"]
    dataset_repo = smol.get("train_dataset_repo_id")
    if not dataset_repo:
        raise ValueError("Set smolvla.train_dataset_repo_id in configs/smolvla.yaml.")
    cmd = [
        "lerobot-train",
        f"--policy.path={smol.get('base_model', 'lerobot/smolvla_base')}",
        f"--dataset.repo_id={dataset_repo}",
        f"--batch_size={int(smol.get('batch_size', 16))}",
        f"--steps={int(smol.get('steps', 20000))}",
        f"--output_dir={smol.get('output_dir', 'outputs/train/smolvla_openx')}",
        f"--job_name={smol.get('job_name', 'smolvla_openx')}",
        f"--policy.device={smol.get('device', 'cuda')}",
        f"--policy.dtype={smol.get('dtype', 'bfloat16')}",
        f"--wandb.enable={str(bool(smol.get('wandb', False))).lower()}",
    ]
    if smol.get("policy_repo_id"):
        cmd.append(f"--policy.repo_id={smol['policy_repo_id']}")
    cmd.extend(extra_args)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _as_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu()
    return torch.tensor(x, dtype=torch.float32)


def _align_action(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    dim = min(pred.numel(), target.numel())
    return pred[:dim], target[:dim]


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def eval_offline(cfg: dict[str, Any], policy_path: str | None) -> None:
    LeRobotDataset, make_pre_post_processors, SmolVLAPolicy = _require_lerobot()
    smol = cfg["smolvla"]
    model_id = policy_path or smol.get("base_model", "lerobot/smolvla_base")
    dataset_repo = smol.get("val_dataset_repo_id")
    if not dataset_repo:
        raise ValueError("Set smolvla.val_dataset_repo_id in configs/smolvla.yaml.")

    device = torch.device(smol.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    dataset = LeRobotDataset(dataset_repo)

    total_mse = 0.0
    total_mae = 0.0
    count = 0
    max_samples = min(int(smol.get("max_eval_samples", 1024)), len(dataset))
    for idx in tqdm(range(max_samples), desc="eval smolvla"):
        frame = dict(dataset[idx])
        batch = preprocess(frame)
        try:
            pred = policy.select_action(batch)
        except Exception:
            pred = policy.select_action(frame)
        pred = postprocess(pred)
        pred_t, target_t = _align_action(_as_tensor(pred), _as_tensor(frame["action"]))
        total_mse += float((pred_t - target_t).pow(2).mean())
        total_mae += float((pred_t - target_t).abs().mean())
        count += 1

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline": "smolvla_zero_shot" if policy_path is None else "smolvla_finetuned",
        "checkpoint": model_id,
        "seed": cfg.get("seed", 42),
        "mse": total_mse / max(count, 1),
        "mae": total_mae / max(count, 1),
        "pred_temporal_smoothness": float("nan"),
        "success_rate": float("nan"),
        "rollout_consistency": float("nan"),
    }
    append_result(Path(smol.get("results_path", "results/baselines.csv")), row)
    print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolVLA external baseline wrapper for LeRobot datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, default=Path("configs/smolvla.yaml"))
    train_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--config", type=Path, default=Path("configs/smolvla.yaml"))
    eval_parser.add_argument("--policy-path", type=str, default=None)

    args = parser.parse_args()
    cfg = _load_config(args.config)
    if args.command == "train":
        train(cfg, args.extra_args)
    elif args.command == "eval":
        eval_offline(cfg, args.policy_path)


if __name__ == "__main__":
    main()
