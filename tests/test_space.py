"""Demo logic and the Space bundle, with a fake model. No Gradio, network or GGUF needed."""

import re
from types import SimpleNamespace
from unittest.mock import patch

import config
from app import space
from scripts import push_space


class FakeModel:
    def __init__(self, content: str = "fix(auth): handle a missing session") -> None:
        self.content = content

    def create_chat_completion(self, **params) -> dict:
        self.params = params
        return {"choices": [{"message": {"content": self.content}}]}


def test_generate_returns_message_and_time() -> None:
    message, status = space.generate(space.EXAMPLE_DIFFS[0], FakeModel())
    assert message == "fix(auth): handle a missing session"
    assert re.fullmatch(r"Generated in \d+\.\d s\.", status)


def test_generate_uses_shared_prompt_and_settings() -> None:
    model = FakeModel()
    space.generate(space.EXAMPLE_DIFFS[0], model)
    assert model.params["temperature"] == config.TEMPERATURE
    assert model.params["max_tokens"] == config.MAX_NEW_TOKENS
    assert model.params["repeat_penalty"] == 1.0
    prompt = model.params["messages"][1]["content"]
    assert model.params["messages"][0]["content"] == config.SYSTEM_PROMPT
    assert "index <HASH>..<HASH> 100644" in prompt
    assert "3f2a91c" not in prompt


def test_empty_input_does_not_load_the_model() -> None:
    with patch.object(space, "load_model") as load_model:
        assert space.generate("   ") == ("", "Paste a diff first.")
    load_model.assert_not_called()


def test_long_diff_says_it_was_cut() -> None:
    _, status = space.generate("x" * (config.MAX_PROMPT_DIFF_CHARS + 1), FakeModel())
    assert "Only the first" in status


def test_multi_line_reply_keeps_first_line() -> None:
    assert space.generate("diff", FakeModel("fix: a\n\nmore"))[0] == "fix: a"


def test_three_examples_are_diffs() -> None:
    assert len(space.EXAMPLE_DIFFS) == 3
    assert all(example.startswith("diff --git") for example in space.EXAMPLE_DIFFS)


def test_space_bundle_files_exist() -> None:
    assert all(path.exists() for path in push_space.space_files().values())


def test_space_readme_has_the_required_header() -> None:
    text = (config.ROOT_DIR / "spaces" / "README.md").read_text(encoding="utf-8")
    header = text.split("---")[1]
    assert "sdk: gradio" in header
    assert "app_file: app.py" in header
    assert "license: cc-by-nc-4.0" in header


def test_dry_run_uploads_nothing(capsys) -> None:
    with patch("sys.argv", ["push_space"]), patch("huggingface_hub.HfApi") as api:
        push_space.main()
    api.assert_not_called()
    assert "Nothing was uploaded" in capsys.readouterr().out


def test_push_uploads_space_and_model_card(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-value")
    client = SimpleNamespace(
        create_repo=lambda *args, **kwargs: None,
        create_commit=lambda *args, **kwargs: None,
        upload_file=lambda **kwargs: uploads.append(kwargs),
    )
    uploads: list[dict] = []
    with (
        patch("sys.argv", ["push_space", "--push"]),
        patch("huggingface_hub.HfApi", lambda token: client),
    ):
        push_space.main()
    assert uploads[0]["repo_id"] == config.ADAPTER_REPO
    assert uploads[0]["path_in_repo"] == "README.md"


def test_model_card_covers_the_required_sections() -> None:
    text = (config.ROOT_DIR / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8")
    sections = (
        "Intended use",
        "Use with Ollama",
        "Data",
        "Training",
        "Results",
        "Limitations",
        "License",
    )
    for section in sections:
        assert f"\n## {section}\n" in text
    assert "CC BY-NC 4.0" in text
