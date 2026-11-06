"""Each filter in data/prepare.py, tested on hand-made rows."""

from unittest.mock import patch

from data import prepare

HEADER = "diff --git a/a.py b/a.py\nindex <HASH>..<HASH> 100644\n--- a/a.py\n+++ b/a.py\n"
GOOD_DIFF = HEADER + "@@ -1 +1 @@\n" + "+x = 1\n" * 40
GOOD_MESSAGE = "Handle expired tokens in the auth layer"


def make_row(**changes) -> dict:
    row = {
        "hash": "abc",
        "diff": GOOD_DIFF,
        "message": GOOD_MESSAGE,
        "project": "owner_repo",
        "split": "train",
        "diff_languages": "py",
    }
    return row | changes


def test_good_row_is_kept() -> None:
    assert prepare.drop_reason(GOOD_DIFF, GOOD_MESSAGE) is None


def test_diff_length_limits() -> None:
    assert prepare.drop_reason(HEADER, GOOD_MESSAGE) == "diff_length"
    assert prepare.drop_reason(GOOD_DIFF + "+y\n" * 2000, GOOD_MESSAGE) == "diff_length"


def test_too_many_files() -> None:
    assert prepare.files_touched(HEADER * 3) == 3
    assert prepare.drop_reason(GOOD_DIFF + HEADER * 3, GOOD_MESSAGE) == "too_many_files"


def test_diff_text_inside_a_line_is_not_a_file() -> None:
    assert prepare.files_touched(HEADER + "+ diff --git a/b b/b\n") == 1


def test_message_length_uses_first_line_only() -> None:
    assert prepare.drop_reason(GOOD_DIFF, "too short") == "message_length"
    assert prepare.drop_reason(GOOD_DIFF, "x" * 101) == "message_length"
    assert prepare.drop_reason(GOOD_DIFF, GOOD_MESSAGE + "\n\n" + "body " * 50) is None


def test_low_value_messages() -> None:
    for message in [
        "minor changes",
        "small fixes and updates",
        "wip wip wip wip",
        "Merge branch 'main' into dev",
        "Merge pull request #12 from a/b",
        "Bump version to 1.2.3",
        "Release 2.0.1 now",
        "v1.2.3-beta.1",
        "fixes #12345",
        "closes #12345",
        "PROJ-123456",
        "Bump version to <I>",
        "Release <I> today",
        "fixes #<I>",
        "version <I>.<I>.<I>",
        "v<I>.<I>.<I>-beta",
    ]:
        assert prepare.is_low_value(message), message


def test_useful_messages_are_not_low_value() -> None:
    for message in [
        "fix race in worker pool",
        "update readme with install steps",
        "release the lock on close",
        "add merge-sort helper for lists",
        "Task #<I>: Block titles",
    ]:
        assert not prepare.is_low_value(message), message


def test_secret_detection() -> None:
    for secret in [
        "+key = AKIAABCDEFGHIJKLMNOP",
        "+-----BEGIN RSA PRIVATE KEY-----",
        '+password = "hunter2hunter2"',
        '+"api_key": "abcd1234efgh"',
        "+token = ghp_" + "a" * 36,
    ]:
        assert prepare.has_secret(secret), secret
    assert not prepare.has_secret('+password = get_password("name")\n+x = 1')


def test_main_language_takes_first_entry() -> None:
    assert prepare.main_language("py, js") == "py"
    assert prepare.main_language("") == "unknown"


def test_output_row_columns() -> None:
    output = prepare.to_output_row(make_row())
    assert set(output) == {"id", "language", "project", "diff", "original_message"}


def test_filter_rows_counts_drops_and_duplicates() -> None:
    rows = [make_row(hash="1"), make_row(hash="1"), make_row(hash="2", message="wip")]
    kept, drops, seen = prepare.filter_rows(rows)
    assert (len(kept), seen) == (1, 3)
    assert drops == {"duplicate": 1, "message_length": 1}


def test_filter_rows_language_cap() -> None:
    rows = [make_row(hash=str(i)) for i in range(3)]
    with patch("config.MAX_ROWS_PER_LANGUAGE", 2):
        kept, drops, _ = prepare.filter_rows(rows)
    assert len(kept) == 2 and drops == {"language_cap": 1}


def test_filter_rows_stops_at_total_cap_and_max_rows() -> None:
    rows = [make_row(hash=str(i)) for i in range(5)]
    with patch("config.MAX_KEPT_ROWS", 2):
        assert len(prepare.filter_rows(rows)[0]) == 2
    assert prepare.filter_rows(rows, max_rows=3)[2] == 3
