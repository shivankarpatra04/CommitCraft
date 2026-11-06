"""Filter labeled rows, split them by project, and push the dataset to the Hugging Face Hub."""

import argparse
import hashlib
import json
import os
from collections import Counter

from rich.console import Console

import config
from data.relabel import keep_label

console = Console()

COLUMNS = ["id", "language", "project", "diff", "original_message", "message", "type", "quality"]

CARD = """---
license: cc-by-nc-4.0
language:
- en
task_categories:
- text-generation
tags:
- git
- commit-messages
---

# CommitCraft data

Git diffs paired with Conventional Commit messages, used to fine-tune
`{base_model}` in the CommitCraft project.

## Source and license

Diffs and original messages come from
[CommitBench](https://huggingface.co/datasets/Maxscha/commitbench). This dataset keeps its
**CC BY-NC 4.0** license: non-commercial use only, credit required.

## How it was made

1. CommitBench rows were filtered: diff size, files touched, low-value messages, secrets.
2. `{claude_model}` wrote a new `message` for each row and gave it a quality score from 1 to 5.
3. Rows were kept only if the message passes the format check and quality is at least {min_quality}.
4. Rows were split by project, so no repository appears in two splits.

## Limitations

- `message` was written by a model, so some labels may be wrong.
- CommitBench hides numbers and hashes as `<I>` and `<HASH>` in the diffs.

## Splits

{split_table}
"""


def split_name(project: str) -> str:
    """Pick a split from a hash of the project name, so a project always lands in one split."""
    bucket = int(hashlib.sha256(project.encode("utf-8")).hexdigest(), 16) % 10_000 / 10_000
    cumulative = 0.0
    for name, ratio in config.SPLIT_RATIOS.items():
        cumulative += ratio
        if bucket < cumulative:
            return name
    return name


def split_by_project(rows: list[dict]) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {name: [] for name in config.SPLIT_RATIOS}
    for row in rows:
        splits[split_name(row["project"])].append(row)
    return splits


def find_project_overlap(splits: dict[str, list[dict]]) -> set[str]:
    """Return every project found in more than one split. It must be empty."""
    counts: Counter = Counter()
    for rows in splits.values():
        counts.update({row["project"] for row in rows})
    return {project for project, count in counts.items() if count > 1}


def load_kept_rows() -> tuple[list[dict], int]:
    """Return rows that pass keep_label, and how many rows were read."""
    with config.LABELED_PATH.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    return [{key: row[key] for key in COLUMNS} for row in rows if keep_label(row)], len(rows)


def build_card(splits: dict[str, list[dict]]) -> str:
    table = "\n".join(f"- {name}: {len(rows):,} rows" for name, rows in splits.items())
    return CARD.format(
        base_model=config.BASE_MODEL,
        claude_model=config.CLAUDE_MODEL,
        min_quality=config.MIN_QUALITY,
        split_table=table,
    )


def push(splits: dict[str, list[dict]]) -> None:
    """Upload the splits and card, then download them again to prove they are correct."""
    from datasets import Dataset, DatasetDict, load_dataset
    from huggingface_hub import DatasetCard

    token = os.environ["HF_TOKEN"]
    data = DatasetDict({name: Dataset.from_list(rows) for name, rows in splits.items()})
    data.push_to_hub(config.DATASET_REPO, token=token)
    DatasetCard(build_card(splits)).push_to_hub(config.DATASET_REPO, token=token)

    hub = load_dataset(config.DATASET_REPO, token=token, download_mode="force_redownload")
    overlap = find_project_overlap({name: list(hub[name]) for name in hub})
    console.print(f"On the Hub: { {name: len(hub[name]) for name in hub} }")
    console.print(f"Projects in more than one split: {len(overlap)}")
    if overlap:
        raise SystemExit("Project overlap found on the Hub")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Upload to the Hub")
    args = parser.parse_args()

    rows, total = load_kept_rows()
    splits = split_by_project(rows)
    console.print(f"Read {total:,} labels, kept {len(rows):,}")
    console.print({name: len(split) for name, split in splits.items()})
    console.print(f"Projects in more than one split: {len(find_project_overlap(splits))}")
    if len(rows) < config.MIN_KEPT_LABELS:
        console.print(f"[yellow]Below the {config.MIN_KEPT_LABELS:,} rows needed[/yellow]")
        if args.push:
            raise SystemExit("Not pushing: too few rows")
    if args.push:
        push(splits)


if __name__ == "__main__":
    main()
