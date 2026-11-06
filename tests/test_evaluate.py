"""Evaluation logic, with fake models and a fake Claude. No GPU, network or API calls."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import config
from eval import evaluate, judge, report

GOOD = "fix(api): handle empty input"
ROWS = [{"diff": f"diff a b\n+x{i}", "message": "fix(a): handle empty input"} for i in range(5)]


def fake_generate(diff: str, use_adapter: bool) -> tuple[str, float]:
    return (GOOD, 0.5) if use_adapter else ("Here is a commit message", 1.5)


class FakeClaude:
    """Always prefers message 1, so results reveal whether the A/B order is really mixed."""

    def __init__(self, winner: str = "1", error: Exception | None = None) -> None:
        self.winner, self.error = winner, error
        self.messages = self

    def create(self, **params) -> SimpleNamespace:
        if self.error:
            raise self.error
        self.last_user = params["messages"][0]["content"]
        text = json.dumps({"score_1": 5, "score_2": 2, "winner": self.winner})
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
        )


def test_format_check() -> None:
    assert report.format_ok(GOOD)
    assert not report.format_ok("Fix the bug")
    assert not report.format_ok(GOOD + "\n\nExtra body text")
    assert not report.format_ok("fix: " + "x" * 80)


@pytest.mark.parametrize(
    ("message", "expected"),
    [("fix(api): x", "fix"), ("docs: x", "docs"), ("fix the bug", None), ("Here: x", None)],
)
def test_commit_type(message: str, expected: str | None) -> None:
    assert report.commit_type(message) == expected


@pytest.mark.parametrize(("swap", "winner"), [(False, "base"), (True, "finetuned")])
def test_judge_maps_position_back_to_the_model(swap: bool, winner: str) -> None:
    client = FakeClaude()
    verdict, cost = judge.judge_pair(client, "diff", "BASE", "TUNED", swap)
    assert verdict["winner"] == winner
    assert verdict[f"score_{winner}"] == 5
    assert cost > 0
    first, second = ("TUNED", "BASE") if swap else ("BASE", "TUNED")
    assert client.last_user.index(first) < client.last_user.index(second)


def test_judge_tie_stays_a_tie() -> None:
    verdict, _ = judge.judge_pair(FakeClaude(winner="tie"), "diff", "A", "B", True)
    assert verdict["winner"] == "tie"


def test_judge_failure_does_not_stop_the_run() -> None:
    assert judge.judge_pair(FakeClaude(error=KeyError("x")), "d", "A", "B", False)[0] is None


def test_ab_order_is_randomised() -> None:
    predictions = [
        {"diff": "d", "base": "A", "finetuned": "B", "reference": "r"} for _ in range(40)
    ]
    judge.judge_all(predictions, FakeClaude())
    swapped = sum(row["judge_swapped"] for row in predictions)
    assert 5 < swapped < 35
    winners = {row["judge"]["winner"] for row in predictions}
    assert winners == {"base", "finetuned"}


def test_limit_5_dry_run_with_mocked_models(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-value")
    predictions_path, results_path = tmp_path / "p.jsonl", tmp_path / "r.md"
    with (
        patch.object(config, "OUTPUTS_DIR", tmp_path),
        patch.object(config, "PREDICTIONS_PATH", predictions_path),
        patch.object(config, "RESULTS_PATH", results_path),
        patch.object(evaluate, "load_rows", lambda limit: ROWS[:limit]),
        patch.object(evaluate, "load_generator", lambda: fake_generate),
        patch.object(evaluate.anthropic, "Anthropic", FakeClaude),
        patch.object(report, "compute_rouge", lambda p, r: 0.5),
    ):
        evaluate.main(["--limit", "5"])

    saved = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    assert len(saved) == 5
    assert saved[0]["finetuned"] == GOOD
    text = results_path.read_text()
    assert "| Format compliance | 0.0% | 100.0% |" in text
    assert "| Correct type | 0.0% | 100.0% |" in text
    assert "| Average latency (GPU) | 1.50 s | 0.50 s |" in text
    assert "Label:" in text


def test_one_line_keeps_a_markdown_example_intact() -> None:
    assert report.one_line("```markdown\nfix: a `b`\n\nmore") == "'''markdown fix: a 'b' more"


def test_missing_key_stops_before_any_gpu_work(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch.object(evaluate, "load_rows") as load_rows, pytest.raises(SystemExit):
        evaluate.main([])
    load_rows.assert_not_called()
