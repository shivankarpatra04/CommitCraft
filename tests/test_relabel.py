"""Relabel and publish logic, with Claude and the Hub mocked."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from data import publish, relabel

GOOD = {"message": "fix(auth): handle expired token", "type": "fix", "quality": 5, "reason": "ok"}


def make_row(row_id: str = "a", project: str = "owner_repo") -> dict:
    return {
        "id": row_id,
        "language": "py",
        "project": project,
        "diff": "diff --git a/a b/a",
        "original_message": "fix token",
    }


def make_message(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


@pytest.fixture
def paths(tmp_path):
    """Send all output files to a temp folder."""
    with (
        patch("config.OUTPUTS_DIR", tmp_path),
        patch("config.LABELED_PATH", tmp_path / "labeled.jsonl"),
        patch("config.BATCH_ID_PATH", tmp_path / "batch_id.txt"),
        patch("config.RAW_PATH", tmp_path / "raw.jsonl"),
    ):
        yield tmp_path


@pytest.mark.parametrize(
    "changes",
    [
        {"quality": 3},
        {"message": "Fixed stuff."},
        {"message": "fix: token at <I> seconds"},
        {"type": "feat"},
        {"message": "fix(auth): " + "x" * 65},
    ],
)
def test_keep_label_drops_bad_labels(changes: dict) -> None:
    assert not relabel.keep_label(GOOD | changes)


def test_keep_label_accepts_good_label() -> None:
    assert relabel.keep_label(GOOD)


def test_parse_label_reads_json() -> None:
    assert relabel.parse_label(make_message(json.dumps(GOOD))) == GOOD


def test_parse_label_turns_broken_reply_into_dropped_label() -> None:
    label = relabel.parse_label(make_message('{"message": "fix', "max_tokens"))
    assert not relabel.keep_label(label)
    assert "max_tokens" in label["reason"]


def test_build_params_puts_diff_in_prompt() -> None:
    params = relabel.build_params(make_row())
    assert "diff --git a/a b/a" in params["messages"][0]["content"]
    assert params["output_config"]["format"]["type"] == "json_schema"


def test_select_pending_is_safe_to_run_twice() -> None:
    rows = [make_row("a"), make_row("b"), make_row("c")]
    assert [row["id"] for row in relabel.select_pending(rows, {"a"}, 2)] == ["b"]
    assert relabel.select_pending(rows, {"a", "b"}, 2) == []


def test_plain_run_saves_each_row_for_resume(paths) -> None:
    client = MagicMock()
    client.messages.create.return_value = make_message(json.dumps(GOOD))
    totals = {"input": 0, "output": 0}
    relabel.run_plain(client, [make_row("a"), make_row("b")], totals)
    assert relabel.load_done_ids() == {"a", "b"}
    assert totals == {"input": 200, "output": 40}


def fake_batch_client() -> MagicMock:
    good = SimpleNamespace(type="succeeded", message=make_message(json.dumps(GOOD)))
    client = MagicMock()
    client.messages.batches.create.return_value = SimpleNamespace(id="batch_1")
    client.messages.batches.retrieve.return_value = SimpleNamespace(processing_status="ended")
    client.messages.batches.results.return_value = [
        SimpleNamespace(custom_id="a", result=good),
        SimpleNamespace(custom_id="b", result=SimpleNamespace(type="errored")),
    ]
    return client


def test_batch_saves_successes_and_retries_failures(paths) -> None:
    rows = [make_row("a"), make_row("b")]
    (paths / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    relabel.run_batch(fake_batch_client(), rows, {"input": 0, "output": 0})
    assert relabel.load_done_ids() == {"a"}
    assert not (paths / "batch_id.txt").exists()


def test_batch_resume_does_not_pay_twice(paths) -> None:
    (paths / "batch_id.txt").write_text("batch_old", encoding="utf-8")
    client = fake_batch_client()
    assert relabel.start_or_resume_batch(client, [make_row()]) == "batch_old"
    client.messages.batches.create.assert_not_called()


def test_split_has_no_project_overlap_and_is_stable() -> None:
    rows = [make_row(str(i), f"project_{i % 300}") for i in range(3000)]
    splits = publish.split_by_project(rows)
    assert publish.find_project_overlap(splits) == set()
    assert sum(len(split) for split in splits.values()) == 3000
    assert all(splits.values())
    assert publish.split_by_project(rows) == splits


def test_split_ratios_are_close_to_plan() -> None:
    splits = publish.split_by_project([make_row(str(i), f"p{i}") for i in range(10000)])
    assert 8800 < len(splits["train"]) < 9200


def test_overlap_check_catches_a_shared_project() -> None:
    splits = {"train": [make_row()], "test": [make_row()]}
    assert publish.find_project_overlap(splits) == {"owner_repo"}


def test_card_credits_commitbench_and_license() -> None:
    card = publish.build_card({"train": [make_row()]})
    assert "cc-by-nc-4.0" in card and "CommitBench" in card and "train: 1 rows" in card
