"""Compare base and fine-tuned models on test rows. Needs a GPU; run it through Kaggle."""

import argparse
import json
import os
import random
import time
from contextlib import nullcontext

import anthropic
from rich.console import Console

import config
from eval import judge, report

console = Console()


def load_rows(limit: int) -> list[dict]:
    """Take a fixed random sample of the test split."""
    from datasets import load_dataset

    rows = list(load_dataset(config.DATASET_REPO)["test"])
    return random.Random(config.EVAL_SEED).sample(rows, min(limit, len(rows)))


def load_generator():
    """Return generate(diff, use_adapter) -> (message, seconds), on one model for both A and B."""
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from train.train import load_base_model

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = PeftModel.from_pretrained(load_base_model(), config.ADAPTER_REPO)
    model.eval()

    def generate(diff: str, use_adapter: bool) -> tuple[str, float]:
        prompt = tokenizer.apply_chat_template(
            config.build_messages(diff), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        adapter = nullcontext() if use_adapter else model.disable_adapter()
        start = time.perf_counter()
        with adapter, torch.inference_mode(), torch.autocast(model.device.type, torch.float16):
            # Clear the model's default sampling settings, so generation is truly greedy.
            output = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
            )
        seconds = time.perf_counter() - start
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), seconds

    return generate


def generate_all(rows: list[dict], generate) -> list[dict]:
    """Generate with both models. The first calls load things into memory, so they are not timed."""
    generate(rows[0]["diff"], False)
    generate(rows[0]["diff"], True)
    predictions = []
    for index, row in enumerate(rows, 1):
        base, base_seconds = generate(row["diff"], False)
        tuned, tuned_seconds = generate(row["diff"], True)
        predictions.append(
            {
                "diff": row["diff"],
                "reference": row["message"],
                "base": base,
                "finetuned": tuned,
                "base_seconds": base_seconds,
                "finetuned_seconds": tuned_seconds,
            }
        )
        console.print(f"Generated {index}/{len(rows)}")
    return predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=config.EVAL_MAX_ROWS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # Fail now, not after the GPU work.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set.")

    rows = load_rows(min(args.limit, config.EVAL_MAX_ROWS))
    predictions = generate_all(rows, load_generator())
    judge.judge_all(predictions, anthropic.Anthropic())

    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    with config.PREDICTIONS_PATH.open("w", encoding="utf-8") as file:
        file.writelines(json.dumps(row) + "\n" for row in predictions)

    base = report.summarize_model(predictions, "base")
    tuned = report.summarize_model(predictions, "finetuned")
    results = report.build_results(predictions, base, tuned)
    config.RESULTS_PATH.write_text(results, encoding="utf-8")
    console.print(results, markup=False, highlight=False)


if __name__ == "__main__":
    main()
