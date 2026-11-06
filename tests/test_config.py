"""Checks for the shared prompt template and commit regex."""

import re

import config


def test_regex_accepts_good_message() -> None:
    assert re.match(config.COMMIT_REGEX, "fix(auth): handle expired token")


def test_regex_rejects_bad_message() -> None:
    assert not re.match(config.COMMIT_REGEX, "fixed stuff.")


def test_user_prompt_keeps_braces_in_diff() -> None:
    diff = "+ if (x) { return {}; }"
    assert config.USER_PROMPT.format(diff=diff).endswith(diff)
