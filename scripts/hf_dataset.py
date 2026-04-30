from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, upload_large_folder


def create_repo(repo_id: str, private: bool) -> None:
    api = HfApi()
    url = api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"Dataset repo ready: {url}")


def upload(repo_id: str, local_dir: Path) -> None:
    upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=local_dir,
    )
    print(f"Uploaded {local_dir} to {repo_id}")


def download(repo_id: str, local_dir: Path) -> None:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["*.tar", "*.json", "README.md"],
    )
    print(f"Downloaded {repo_id} to {local_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/upload/download the Hugging Face Dataset repo for VLA shards.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("repo_id", help="Example: username/vla-franka-subset")
    create_parser.add_argument("--public", action="store_true", help="Create a public dataset repo instead of private.")

    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("repo_id")
    upload_parser.add_argument("--local-dir", type=Path, default=Path("data/webdataset"))

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("repo_id")
    download_parser.add_argument("--local-dir", type=Path, default=Path("data/webdataset"))

    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    if args.command == "create":
        create_repo(args.repo_id, private=not args.public)
    elif args.command == "upload":
        upload(args.repo_id, args.local_dir)
    elif args.command == "download":
        download(args.repo_id, args.local_dir)


if __name__ == "__main__":
    main()
