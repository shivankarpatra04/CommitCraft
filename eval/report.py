"""Metrics for each model and the markdown report."""

import re

import config

TYPE_REGEX = rf"^({'|'.join(config.COMMIT_TYPES)})(\([^)]*\))?:"


def format_ok(message: str) -> bool:
    """Same format rule the training labels had to pass."""
    return (
        re.match(config.COMMIT_REGEX, message) is not None
        and len(message) <= config.MAX_COMMIT_CHARS
    )


def commit_type(message: str) -> str | None:
    match = re.match(TYPE_REGEX, message)
    return match.group(1) if match else None


def compute_rouge(predictions: list[str], references: list[str]) -> float:
    import evaluate

    return evaluate.load("rouge").compute(predictions=predictions, references=references)["rougeL"]


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_model(predictions: list[dict], name: str) -> dict:
    """Compute every metric for one model."""
    messages = [row[name] for row in predictions]
    references = [row["reference"] for row in predictions]
    types = zip(messages, references, strict=True)
    verdicts = [row["judge"] for row in predictions if row["judge"]]
    return {
        "format": 100 * average([format_ok(message) for message in messages]),
        "type": 100 * average([commit_type(m) == commit_type(r) for m, r in types]),
        "rouge": compute_rouge(messages, references),
        "judge": average([verdict[f"score_{name}"] for verdict in verdicts]),
        "seconds": average([row[f"{name}_seconds"] for row in predictions]),
    }


def one_line(text: str) -> str:
    """Make model output safe to show inside one markdown code span."""
    return " ".join(text.split()).replace("`", "'")[:200]


def build_results(predictions: list[dict], base: dict, tuned: dict) -> str:
    """Write the markdown report: metrics table, verdict and the largest judge-score gaps."""
    verdicts = [row["judge"] for row in predictions if row["judge"]]
    wins = {name: sum(v["winner"] == name for v in verdicts) for name in ("base", "finetuned")}
    ties = len(verdicts) - sum(wins.values())
    win_rate = 100 * wins["finetuned"] / len(verdicts) if verdicts else 0.0
    beats = tuned["format"] > base["format"] and tuned["judge"] > base["judge"]
    lines = [
        "# CommitCraft evaluation",
        "",
        f"{len(predictions)} test rows, greedy, at most {config.MAX_NEW_TOKENS} new tokens. "
        f"Judge: `{config.CLAUDE_MODEL}`, {len(verdicts)} verdicts.",
        "",
        "| Metric | A: base | B: fine-tuned |",
        "| --- | --- | --- |",
        f"| Format compliance | {base['format']:.1f}% | {tuned['format']:.1f}% |",
        f"| Correct type | {base['type']:.1f}% | {tuned['type']:.1f}% |",
        f"| ROUGE-L | {base['rouge']:.3f} | {tuned['rouge']:.3f} |",
        f"| Judge score (1-5) | {base['judge']:.2f} | {tuned['judge']:.2f} |",
        f"| Average latency (GPU) | {base['seconds']:.2f} s | {tuned['seconds']:.2f} s |",
        "",
        f"B win rate: {win_rate:.1f}% (B wins {wins['finetuned']}, A wins {wins['base']}, "
        f"ties {ties}).",
        f"B beats A on format and judge score: {'yes' if beats else 'no'}.",
        "",
        "## Largest judge-score gaps in favour of B",
        "",
        "These are chosen as the best cases, so they are not typical.",
    ]
    judged = [row for row in predictions if row["judge"]]
    judged.sort(key=lambda r: r["judge"]["score_base"] - r["judge"]["score_finetuned"])
    for row in judged[: config.BEST_EXAMPLES]:
        lines += [
            "",
            f"- A ({row['judge']['score_base']}/5): `{one_line(row['base'])}`",
            f"- B ({row['judge']['score_finetuned']}/5): `{one_line(row['finetuned'])}`",
            f"- Label: `{one_line(row['reference'])}`",
        ]
    return "\n".join(lines) + "\n"
