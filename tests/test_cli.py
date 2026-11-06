"""CLI and git hook, with real git in a temp repo and Ollama mocked. No network or GPU."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from typer.testing import CliRunner

import config
from app import cli

runner = CliRunner()
MESSAGE = "fix(app): handle empty input"


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def fake_reply(content: str = MESSAGE, status: int = 200) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=status,
        ok=status == 200,
        json=lambda: {"message": {"content": content}},
    )


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp repo with one commit and one staged change."""
    path = tmp_path / "repo"
    path.mkdir()
    monkeypatch.chdir(path)
    for name in ("GIT_AUTHOR", "GIT_COMMITTER"):
        monkeypatch.setenv(f"{name}_NAME", "Test")
        monkeypatch.setenv(f"{name}_EMAIL", "test@example.com")
    git("init", "-q")
    (path / "a.txt").write_text("one\n")
    git("add", ".")
    git("commit", "-q", "-m", "chore: first commit")
    (path / "a.txt").write_text("two\n")
    git("add", ".")
    return path


def test_diff_matches_the_training_format(repo: Path) -> None:
    diff = cli.staged_diff()
    assert diff.startswith("diff --git a/a.txt b/a.txt\nindex <HASH>..<HASH> 100644\n")
    assert "+two" in diff


def test_suggest_sends_the_shared_prompt(repo: Path) -> None:
    with patch.object(cli.requests, "post", return_value=fake_reply()) as post:
        result = runner.invoke(cli.app, ["suggest"])
    assert result.exit_code == 0
    assert result.stdout.strip() == MESSAGE
    body = post.call_args.kwargs["json"]
    assert body["model"] == config.OLLAMA_MODEL
    assert body["messages"] == config.build_messages(cli.staged_diff())
    assert body["options"] == {"temperature": 0.2, "num_predict": 40}


def test_long_diff_is_cut_to_the_training_length(repo: Path) -> None:
    (repo / "a.txt").write_text("x\n" * 5000)
    git("add", ".")
    with patch.object(cli.requests, "post", return_value=fake_reply()) as post:
        runner.invoke(cli.app, ["suggest"])
    prompt = post.call_args.kwargs["json"]["messages"][1]["content"]
    assert len(prompt) <= len(config.USER_PROMPT) + config.MAX_PROMPT_DIFF_CHARS


def test_only_the_first_line_is_used(repo: Path) -> None:
    with patch.object(cli.requests, "post", return_value=fake_reply(f"{MESSAGE}\n\nmore text")):
        assert runner.invoke(cli.app, ["suggest"]).stdout.strip() == MESSAGE


def test_no_staged_changes(repo: Path) -> None:
    git("commit", "-q", "-m", "fix: second commit")
    result = runner.invoke(cli.app, ["suggest"])
    assert result.exit_code == 1
    assert "No staged changes" in result.stderr


@pytest.mark.parametrize(
    ("mock", "expected"),
    [
        ({"side_effect": requests.ConnectionError()}, "Ollama is not running"),
        ({"side_effect": requests.Timeout()}, "did not reply"),
        ({"return_value": fake_reply(status=404)}, "ollama create commitcraft"),
        ({"return_value": fake_reply(status=500)}, "HTTP 500"),
        ({"return_value": fake_reply(content="  ")}, "empty message"),
    ],
)
def test_ollama_errors_are_clear(repo: Path, mock: dict, expected: str) -> None:
    with patch.object(cli.requests, "post", **mock):
        result = runner.invoke(cli.app, ["suggest"])
    assert result.exit_code == 1
    assert expected in result.stderr


def test_outside_a_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    result = runner.invoke(cli.app, ["suggest"])
    assert result.exit_code == 1
    assert "not a git repository" in result.stderr


def test_commit_accept(repo: Path) -> None:
    with patch.object(cli.requests, "post", return_value=fake_reply()):
        assert runner.invoke(cli.app, ["commit"], input="a\n").exit_code == 0
    assert git("log", "-1", "--format=%s") == MESSAGE


def test_commit_edit(repo: Path) -> None:
    with patch.object(cli.requests, "post", return_value=fake_reply()):
        runner.invoke(cli.app, ["commit"], input="e\nfix: my own words\na\n")
    assert git("log", "-1", "--format=%s") == "fix: my own words"


def test_commit_retry(repo: Path) -> None:
    replies = [fake_reply("fix: first try"), fake_reply("fix: second try")]
    with patch.object(cli.requests, "post", side_effect=replies):
        runner.invoke(cli.app, ["commit"], input="r\na\n")
    assert git("log", "-1", "--format=%s") == "fix: second try"


def test_commit_quit_commits_nothing(repo: Path) -> None:
    with patch.object(cli.requests, "post", return_value=fake_reply()):
        assert runner.invoke(cli.app, ["commit"], input="q\n").exit_code == 0
    assert git("log", "-1", "--format=%s") == "chore: first commit"


@pytest.fixture
def hooked(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install the hook, with a fake `commitcraft` command on PATH."""
    assert runner.invoke(cli.app, ["install-hook"]).exit_code == 0
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "commitcraft"
    fake.write_text(f"#!/bin/sh\necho '{MESSAGE}'\n", newline="\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("GIT_EDITOR", "true")
    return repo / ".git" / "hooks" / "prepare-commit-msg"


def test_hook_fills_a_plain_commit(hooked: Path) -> None:
    git("commit", "-q")
    assert git("log", "-1", "--format=%s") == MESSAGE


def test_hook_skips_dash_m(hooked: Path) -> None:
    git("commit", "-q", "-m", "fix: my own message")
    assert git("log", "-1", "--format=%s") == "fix: my own message"


def test_hook_skips_amend(hooked: Path) -> None:
    git("commit", "-q", "-m", "fix: my own message")
    git("commit", "-q", "--amend")
    assert git("log", "-1", "--format=%s") == "fix: my own message"


def test_hook_skips_merge(hooked: Path, repo: Path) -> None:
    message_file = repo / "message.txt"
    message_file.write_text("Merge branch 'x'\n")
    subprocess.run(["sh", str(hooked), str(message_file), "merge"], check=True)
    assert message_file.read_text() == "Merge branch 'x'\n"


def test_hook_never_blocks_when_suggest_fails(hooked: Path, repo: Path, tmp_path: Path) -> None:
    (tmp_path / "bin" / "commitcraft").write_text("#!/bin/sh\nexit 1\n", newline="\n")
    message_file = repo / "message.txt"
    message_file.write_text("# comment\n")
    subprocess.run(["sh", str(hooked), str(message_file)], check=True)
    assert message_file.read_text() == "# comment\n"


def test_install_hook_keeps_a_foreign_hook(repo: Path) -> None:
    hook = repo / ".git" / "hooks" / "prepare-commit-msg"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text("#!/bin/sh\necho mine\n")
    result = runner.invoke(cli.app, ["install-hook"])
    assert result.exit_code == 1
    assert "already exists" in result.stderr
    assert "mine" in hook.read_text()


def test_modelfile_matches_the_training_prompt() -> None:
    text = (config.ROOT_DIR / "Modelfile").read_text(encoding="utf-8")
    assert f"FROM ./{config.GGUF_FILENAME}" in text
    assert f'SYSTEM """{config.SYSTEM_PROMPT}"""' in text
    assert f"PARAMETER temperature {config.TEMPERATURE}" in text
    assert f"PARAMETER num_predict {config.MAX_NEW_TOKENS}" in text
    template = text.split('TEMPLATE """')[1].split('"""')[0]
    rendered = template.replace("{{ .System }}", "SYS").replace("{{ .Prompt }}", "USER")
    # Copied from the real tokenizer's apply_chat_template output.
    expected = "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nUSER<|im_end|>\n"
    assert rendered == expected + "<|im_start|>assistant\n"
