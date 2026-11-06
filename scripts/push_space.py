"""Push the model card and optionally the demo Space. Nothing uploads without a flag."""

import argparse
import os
from pathlib import Path

from rich.console import Console

import config

console = Console()


def space_files() -> dict[str, Path]:
    """Map each path in the Space to its local file."""
    return {
        "app.py": config.ROOT_DIR / "spaces" / "app.py",
        "README.md": config.ROOT_DIR / "spaces" / "README.md",
        "requirements.txt": config.ROOT_DIR / "spaces" / "requirements.txt",
        "space.py": config.ROOT_DIR / "app" / "space.py",
        "config.py": config.ROOT_DIR / "config.py",
    }


def push_card() -> None:
    """Upload the model card, which is public from this moment."""
    from huggingface_hub import HfApi

    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj=str(config.ROOT_DIR / "docs" / "MODEL_CARD.md"),
        path_in_repo="README.md",
        repo_id=config.ADAPTER_REPO,
        commit_message="Add model card",
    )
    console.print(f"Model card: https://huggingface.co/{config.ADAPTER_REPO}")


def push_space() -> None:
    """Upload the demo. A Gradio Space on free CPU needs a Hugging Face PRO subscription."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(config.SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
        for name, path in space_files().items()
    ]
    api.create_commit(
        config.SPACE_REPO, operations, commit_message="Update demo", repo_type="space"
    )
    console.print(f"Space: https://huggingface.co/spaces/{config.SPACE_REPO}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", action="store_true", help="Upload the model card")
    parser.add_argument("--space", action="store_true", help="Upload the demo Space")
    args = parser.parse_args()
    if args.card:
        push_card()
    if args.space:
        push_space()
    if args.card or args.space:
        return
    for name, path in space_files().items():
        console.print(f"{config.SPACE_REPO}/{name} <- {path.relative_to(config.ROOT_DIR)}")
    console.print(f"{config.ADAPTER_REPO}/README.md <- docs/MODEL_CARD.md")
    console.print("Dry run. Nothing was uploaded.")


if __name__ == "__main__":
    main()
