"""Merge the adapter into the base model, and upload the finished GGUF. Used by export_gguf.sh."""

import argparse
import os

from rich.console import Console

import config

console = Console()


def merge() -> None:
    """Save base + adapter as one normal model. This runs on CPU, in full 32-bit precision."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, config.ADAPTER_REPO).merge_and_unload()
    merged.save_pretrained(str(config.MERGED_DIR))
    AutoTokenizer.from_pretrained(config.BASE_MODEL).save_pretrained(str(config.MERGED_DIR))
    console.print(f"Merged model saved to {config.MERGED_DIR}")


def upload() -> None:
    """Upload the GGUF next to the adapter, then confirm that the Hub has it."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(
        path_or_fileobj=str(config.EXPORT_DIR / config.GGUF_FILENAME),
        path_in_repo=config.GGUF_FILENAME,
        repo_id=config.ADAPTER_REPO,
    )
    if config.GGUF_FILENAME not in api.list_repo_files(config.ADAPTER_REPO):
        raise SystemExit(f"{config.GGUF_FILENAME} is missing from {config.ADAPTER_REPO}.")
    console.print(f"Uploaded {config.GGUF_FILENAME} to {config.ADAPTER_REPO}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["merge", "upload"])
    {"merge": merge, "upload": upload}[parser.parse_args().step]()


if __name__ == "__main__":
    main()
