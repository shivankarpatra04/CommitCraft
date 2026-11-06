"""Claude compares two commit messages for one diff, without knowing which model wrote them."""

import json
import random

import anthropic
from rich.console import Console

import config

console = Console()

SCORE = {"type": "integer", "enum": [1, 2, 3, 4, 5]}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score_1": SCORE,
        "score_2": SCORE,
        "winner": {"type": "string", "enum": ["1", "2", "tie"]},
    },
    "required": ["score_1", "score_2", "winner"],
    "additionalProperties": False,
}


def judge_pair(client, diff: str, base: str, tuned: str, swap: bool) -> tuple[dict | None, int]:
    """Ask Claude to compare one pair blind. Returns the verdict and the API cost in micro-USD."""
    first, second = (tuned, base) if swap else (base, tuned)
    user = (
        f"<diff>\n{diff[:config.MAX_PROMPT_DIFF_CHARS]}\n</diff>\n"
        f"<message_1>\n{first}\n</message_1>\n<message_2>\n{second}\n</message_2>"
    )
    try:
        reply = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=config.JUDGE_MAX_TOKENS,
            system=config.JUDGE_SYSTEM,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": JUDGE_SCHEMA},
            },
            messages=[{"role": "user", "content": user}],
        )
        text = next(block.text for block in reply.content if block.type == "text")
        verdict = json.loads(text)
        names = {"1": "finetuned" if swap else "base", "2": "base" if swap else "finetuned"}
        cost = (
            reply.usage.input_tokens * config.CLAUDE_INPUT_USD_PER_MTOK
            + reply.usage.output_tokens * config.CLAUDE_OUTPUT_USD_PER_MTOK
        )
        return {
            f"score_{names['1']}": verdict["score_1"],
            f"score_{names['2']}": verdict["score_2"],
            "winner": names.get(verdict["winner"], "tie"),
        }, int(cost)
    except (anthropic.APIError, StopIteration, json.JSONDecodeError, KeyError) as error:
        # One failed verdict must not throw away a finished GPU run.
        console.print(f"[yellow]Judge failed: {type(error).__name__}[/yellow]")
        return None, 0


def judge_all(predictions: list[dict], client) -> None:
    """Judge every pair, with a random A/B order per row so position cannot favour a model."""
    rng = random.Random(config.EVAL_SEED)
    total_cost = 0
    for index, row in enumerate(predictions, 1):
        row["judge_swapped"] = rng.random() < 0.5
        row["judge"], cost = judge_pair(
            client, row["diff"], row["base"], row["finetuned"], row["judge_swapped"]
        )
        total_cost += cost
        console.print(f"Judged {index}/{len(predictions)}")
    console.print(f"Judge cost: ${total_cost / 1_000_000:.4f}")
