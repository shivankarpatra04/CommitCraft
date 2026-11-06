"""Ask Claude to write a clean commit message for each row. Results go to labeled.jsonl."""

import argparse
import json
import random
import re
import time

import anthropic
from rich.console import Console

import config

console = Console()

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "type": {"type": "string", "enum": config.COMMIT_TYPES},
        "quality": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["message", "type", "quality", "reason"],
    "additionalProperties": False,
}


def load_rows() -> list[dict]:
    """Load raw rows in a fixed shuffled order, so every language appears early."""
    with config.RAW_PATH.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    random.Random(config.RELABEL_SEED).shuffle(rows)
    return rows


def load_done_ids() -> set[str]:
    """Return ids already in labeled.jsonl, so we never pay for a row twice."""
    if not config.LABELED_PATH.exists():
        return set()
    with config.LABELED_PATH.open(encoding="utf-8") as file:
        return {json.loads(line)["id"] for line in file}


def select_pending(rows: list[dict], done_ids: set[str], limit: int) -> list[dict]:
    """Take the first `limit` rows and skip finished ones. Running it twice costs nothing."""
    return [row for row in rows[:limit] if row["id"] not in done_ids]


def build_params(row: dict) -> dict:
    """Build the request for one row. Used by plain calls and by batches."""
    user = (
        f"<original_message>\n{row['original_message']}\n</original_message>\n"
        f"<diff>\n{row['diff']}\n</diff>"
    )
    return {
        "model": config.CLAUDE_MODEL,
        "max_tokens": config.RELABEL_MAX_TOKENS,
        "system": config.RELABEL_SYSTEM,
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": LABEL_SCHEMA},
        },
        "messages": [{"role": "user", "content": user}],
    }


def parse_label(message) -> dict:
    """Read Claude's JSON. A refusal or broken reply becomes a label that later gets dropped."""
    text = next((block.text for block in message.content if block.type == "text"), "")
    try:
        label = json.loads(text)
        return {key: label[key] for key in ("message", "type", "quality", "reason")}
    except (json.JSONDecodeError, KeyError, TypeError):
        reason = f"no valid json, stop_reason={message.stop_reason}"
        return {"message": "", "type": "", "quality": 0, "reason": reason}


def keep_label(label: dict) -> bool:
    """A label is kept only if it has the right format and Claude trusts it."""
    message = label["message"]
    return (
        re.match(config.COMMIT_REGEX, message) is not None
        and len(message) <= config.MAX_COMMIT_CHARS
        and message.startswith(label["type"])
        and "<" not in message
        and label["quality"] >= config.MIN_QUALITY
    )


def add_usage(totals: dict, message) -> None:
    totals["input"] += message.usage.input_tokens
    totals["output"] += message.usage.output_tokens


def save_label(row: dict, label: dict) -> None:
    """Append right away, so a crash loses nothing. Dropped labels are saved too (we paid)."""
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    with config.LABELED_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row | label, ensure_ascii=False) + "\n")


def run_plain(client, rows: list[dict], totals: dict) -> list[dict]:
    """Label rows one by one."""
    labels = []
    for index, row in enumerate(rows, start=1):
        message = client.messages.create(**build_params(row))
        add_usage(totals, message)
        label = parse_label(message)
        save_label(row, label)
        labels.append(label)
        console.print(f"[{index}/{len(rows)}] q={label['quality']} {label['message']}")
    return labels


def start_or_resume_batch(client, rows: list[dict]) -> str:
    """Reuse a batch we already started, so a restart never pays for a second one."""
    if config.BATCH_ID_PATH.exists():
        return config.BATCH_ID_PATH.read_text(encoding="utf-8").strip()
    requests = [{"custom_id": row["id"], "params": build_params(row)} for row in rows]
    batch = client.messages.batches.create(requests=requests)
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    config.BATCH_ID_PATH.write_text(batch.id, encoding="utf-8")
    return batch.id


def run_batch(client, rows: list[dict], totals: dict) -> list[dict]:
    """Label rows at half price. Failed rows are not saved, so the next run retries them."""
    batch_id = start_or_resume_batch(client, rows)
    while (batch := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
        console.print(f"batch {batch_id}: {batch.request_counts.processing} still processing")
        time.sleep(config.BATCH_POLL_SECONDS)

    rows_by_id = {row["id"]: row for row in load_rows()}
    done_ids = load_done_ids()
    labels = []
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded" or result.custom_id in done_ids:
            continue
        add_usage(totals, result.result.message)
        label = parse_label(result.result.message)
        save_label(rows_by_id[result.custom_id], label)
        labels.append(label)
    config.BATCH_ID_PATH.unlink()
    return labels


def print_report(labels: list[dict], totals: dict, is_batch: bool) -> None:
    """Show samples, keep rate, token usage and the estimated cost of the full run."""
    for label in labels[:10]:
        console.print(f"q={label['quality']} {label['message']}\n   why: {label['reason']}")
    kept = sum(keep_label(label) for label in labels)
    console.print(f"Labeled {len(labels)}, kept {kept}. Tokens: {totals}")
    if not labels:
        return
    usd = (
        totals["input"] * config.CLAUDE_INPUT_USD_PER_MTOK
        + totals["output"] * config.CLAUDE_OUTPUT_USD_PER_MTOK
    ) / 1_000_000
    if is_batch:
        usd *= config.BATCH_DISCOUNT
    full = usd / len(labels) * config.RELABEL_ROWS
    if not is_batch:
        full *= config.BATCH_DISCOUNT
    console.print(f"This run: ${usd:.4f}")
    console.print(f"Estimate for {config.RELABEL_ROWS:,} rows by batch: ${full:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Use plain calls on the first N rows")
    parser.add_argument("--batch", action="store_true", help="Use the Message Batches API")
    args = parser.parse_args()
    if not args.limit and not args.batch:
        parser.error("choose --limit N or --batch")

    limit = args.limit or config.RELABEL_ROWS
    pending = select_pending(load_rows(), load_done_ids(), limit)
    console.print(f"{len(pending)} rows to label with {config.CLAUDE_MODEL}")
    if not pending and not config.BATCH_ID_PATH.exists():
        return

    client = anthropic.Anthropic()
    totals = {"input": 0, "output": 0}
    run = run_batch if args.batch else run_plain
    print_report(run(client, pending, totals), totals, args.batch)


if __name__ == "__main__":
    main()
