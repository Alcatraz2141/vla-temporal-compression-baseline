from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible baseline train/eval sweep.")
    parser.add_argument("--sweep", type=Path, default=Path("configs/baseline_sweep.yaml"))
    parser.add_argument("--config", type=Path, default=None, help="Override the config path in the sweep YAML.")
    parser.add_argument("--baselines", nargs="*", default=None)
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with args.sweep.open("r", encoding="utf-8") as f:
        sweep = yaml.safe_load(f)
    config = args.config or Path(sweep["config"])
    baselines = args.baselines or list(sweep["baselines"])
    do_train = bool(sweep.get("train", True)) if args.train is None else args.train
    do_eval = bool(sweep.get("eval", True)) if args.eval is None else args.eval

    for baseline in baselines:
        if do_train:
            run([sys.executable, "training/train.py", "--config", str(config), "--baseline", baseline], args.dry_run)
        if do_eval:
            run([sys.executable, "evaluation/eval.py", "--config", str(config), "--baseline", baseline], args.dry_run)


if __name__ == "__main__":
    main()
