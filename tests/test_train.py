"""Training data formatting, on 2 rows with a fake tokenizer. No GPU, network or torch needed."""

import config
from train import train

ROWS = [
    {"diff": "diff --git a/a b/a\n+x", "message": "fix(a): handle empty input"},
    {"diff": "diff --git a/b b/b\n+{y}", "message": "docs: explain the setup steps"},
]


class FakeTokenizer:
    """Turns each character into one token id, so positions are easy to check."""

    eos_token = "$"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "".join(f"[{m['role']}]{m['content']}" for m in messages) + "[assistant]"

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


def to_text(ids: list[int]) -> str:
    return "".join(chr(token) for token in ids)


def test_prompt_uses_shared_template() -> None:
    example = train.format_example(ROWS[0], FakeTokenizer())
    text = to_text(example["input_ids"])
    assert config.SYSTEM_PROMPT in text
    assert config.USER_PROMPT.format(diff=ROWS[0]["diff"]) in text
    assert text.endswith("[assistant]" + ROWS[0]["message"] + "$")


def test_only_the_completion_counts_toward_loss() -> None:
    for row in ROWS:
        example = train.format_example(row, FakeTokenizer())
        learned = [label for label in example["labels"] if label != train.IGNORE_INDEX]
        assert to_text(learned) == row["message"] + "$"
        assert example["labels"][0] == train.IGNORE_INDEX
        assert example["labels"][-len(learned) :] == example["input_ids"][-len(learned) :]


def test_all_lists_have_the_same_length() -> None:
    example = train.format_example(ROWS[1], FakeTokenizer())
    assert len(example["input_ids"]) == len(example["labels"]) == len(example["attention_mask"])


def test_long_diff_is_cut_but_label_survives() -> None:
    long_diff = "x" * (config.MAX_PROMPT_DIFF_CHARS + 500)
    row = {"diff": long_diff, "message": "fix: keep the label safe"}
    text = to_text(train.format_example(row, FakeTokenizer())["input_ids"])
    assert text.count("x") <= config.MAX_PROMPT_DIFF_CHARS + 5
    assert text.endswith(row["message"] + "$")


def test_parser_defaults_and_flags() -> None:
    defaults = train.build_parser().parse_args([])
    assert (defaults.epochs, defaults.max_steps, defaults.push) == (config.EPOCHS, -1, False)
    args = train.build_parser().parse_args(["--max-steps", "20", "--epochs", "1", "--push"])
    assert (args.epochs, args.max_steps, args.push) == (1, 20, True)
