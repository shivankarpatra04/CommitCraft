"""Stream CommitBench, keep useful rows, and save them to outputs/raw.jsonl."""

import argparse
import json
import re
from collections import Counter

from rich.console import Console
from rich.table import Table

import config

console = Console()


def first_line(message: str) -> str:
    """Return the subject line of a commit message."""
    lines = message.strip().splitlines()
    return lines[0].strip() if lines else ""


def diff_length_ok(diff: str) -> bool:
    """Very small diffs say little; very big ones do not fit in training."""
    return config.MIN_DIFF_CHARS <= len(diff) <= config.MAX_DIFF_CHARS


def files_touched(diff: str) -> int:
    """Count files by their `diff --git` header lines."""
    return len(re.findall(r"^diff --git ", diff, flags=re.MULTILINE))


def files_ok(diff: str) -> bool:
    return 1 <= files_touched(diff) <= config.MAX_FILES_TOUCHED


def message_length_ok(message: str) -> bool:
    return config.MIN_MESSAGE_CHARS <= len(first_line(message)) <= config.MAX_MESSAGE_CHARS


def is_low_value(message: str) -> bool:
    """True for merges, version bumps, bare issue numbers and messages like 'minor fixes'."""
    # CommitBench replaces numbers with <I>, so turn it back into a number for the regexes.
    line = first_line(message).lower().replace("<i>", "0.0")
    words = re.findall(r"[a-z0-9]+", line)
    if all(word in config.LOW_VALUE_WORDS for word in words):
        return True
    patterns = (config.MERGE_REGEX, config.VERSION_BUMP_REGEX, config.BARE_ISSUE_REGEX)
    return any(re.search(pattern, line) for pattern in patterns)


def has_secret(diff: str) -> bool:
    """True if the diff contains something that looks like a key or password."""
    return any(re.search(pattern, diff) for pattern in config.SECRET_REGEXES)


def drop_reason(diff: str, message: str) -> str | None:
    """Return the name of the first failed filter, or None if the row is good."""
    if not diff_length_ok(diff):
        return "diff_length"
    if not files_ok(diff):
        return "too_many_files"
    if not message_length_ok(message):
        return "message_length"
    if is_low_value(message):
        return "low_value_message"
    if has_secret(diff):
        return "secret"
    return None


def main_language(diff_languages: str) -> str:
    """CommitBench can list several languages; use the first as the row's language."""
    return diff_languages.split(",")[0].strip() or "unknown"


def to_output_row(row: dict) -> dict:
    """Rename CommitBench columns to the names used in this project."""
    return {
        "id": row["hash"],
        "language": main_language(row["diff_languages"]),
        "project": row["project"],
        "diff": row["diff"],
        "original_message": row["message"].strip(),
    }


def filter_rows(rows, max_rows: int | None = None) -> tuple[list[dict], Counter, int]:
    """Apply all filters and caps. Returns kept rows, drop counts and rows seen."""
    kept: list[dict] = []
    drops: Counter = Counter()
    per_language: Counter = Counter()
    seen_ids: set[str] = set()
    seen = 0
    for row in rows:
        if max_rows is not None and seen >= max_rows:
            break
        if len(kept) >= config.MAX_KEPT_ROWS:
            break
        seen += 1
        if seen % 10000 == 0:
            console.print(f"seen {seen:,} | kept {len(kept):,}")
        language = main_language(row["diff_languages"])
        reason = drop_reason(row["diff"], row["message"])
        if reason is None and row["hash"] in seen_ids:
            reason = "duplicate"
        if reason is None and per_language[language] >= config.MAX_ROWS_PER_LANGUAGE:
            reason = "language_cap"
        if reason:
            drops[reason] += 1
            continue
        seen_ids.add(row["hash"])
        per_language[language] += 1
        kept.append(to_output_row(row))
    return kept, drops, seen


def print_summary(kept: list[dict], drops: Counter, seen: int) -> None:
    """Show seen, kept, kept per language and drops per filter."""
    table = Table(title=f"Seen {seen:,} | Kept {len(kept):,}")
    table.add_column("Group")
    table.add_column("Name")
    table.add_column("Rows", justify="right")
    for language, count in Counter(row["language"] for row in kept).most_common():
        table.add_row("kept", language, f"{count:,}")
    for reason, count in drops.most_common():
        table.add_row("dropped", reason, f"{count:,}")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=None, help="Stop after seeing this many")
    args = parser.parse_args()

    from datasets import load_dataset  # slow import, so only load it when running for real

    stream = load_dataset(config.SOURCE_DATASET, split="train", streaming=True)
    kept, drops, seen = filter_rows(stream, args.max_rows)

    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    with config.RAW_PATH.open("w", encoding="utf-8") as file:
        for row in kept:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    console.print(f"Saved {len(kept):,} rows to {config.RAW_PATH}")
    print_summary(kept, drops, seen)


if __name__ == "__main__":
    main()
