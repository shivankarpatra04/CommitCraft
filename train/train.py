"""Fine-tune the base model with QLoRA. Needs a GPU; run it through the Kaggle notebook."""

import argparse
import os

from rich.console import Console

import config

console = Console()

# Hugging Face losses skip any position whose label is -100.
IGNORE_INDEX = -100


def format_example(row: dict, tokenizer) -> dict:
    """Tokenize one row so only the commit message counts toward the loss (completion-only)."""
    prompt = tokenizer.apply_chat_template(
        config.build_messages(row["diff"]), tokenize=False, add_generation_prompt=True
    )
    # The end token is part of the target, so the model learns when to stop writing.
    completion = row["message"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + completion_ids,
    }


def tokenize_split(split, tokenizer):
    """Tokenize a split and drop rows too long to fit, since cutting them would cut the label."""
    columns = split.column_names
    tokenized = split.map(lambda row: format_example(row, tokenizer), remove_columns=columns)
    kept = tokenized.filter(lambda row: len(row["input_ids"]) <= config.MAX_SEQ_LENGTH)
    console.print(f"{len(kept):,} rows kept of {len(tokenized):,}")
    return kept


def load_quantized_model():
    """Load the base model in 4-bit and make it ready for LoRA."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        # Store weights in 4 bits instead of 16, so the model fits on a free GPU.
        load_in_4bit=True,
        # nf4 is a 4-bit format designed for how neural network weights are distributed.
        bnb_4bit_quant_type="nf4",
        # Double quant also compresses the quantization data, saving a little more memory.
        bnb_4bit_use_double_quant=True,
        # Weights are stored in 4 bits, but the math runs in 16 bits for accuracy.
        bnb_4bit_compute_dtype=torch.float16,
    )
    # One GPU is plenty for a 4-bit 1.5B model, and it keeps the setup simple.
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, quantization_config=quantization, device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        # r is the size of the small adapter matrices: bigger learns more but uses more memory.
        r=config.LORA_R,
        # alpha scales the adapter's effect; the actual strength is alpha / r = 2.
        lora_alpha=config.LORA_ALPHA,
        # Dropout randomly turns off 5% of adapter values, which reduces overfitting.
        lora_dropout=config.LORA_DROPOUT,
        # Add an adapter to every linear layer except the output head.
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def build_trainer(model, tokenizer, train_data, eval_data, args, report_to: str, use_fp16: bool):
    """Create the trainer. Kept apart from model loading so it can be tested without a GPU."""
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        output_dir=str(config.CHECKPOINTS_DIR),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type="cosine",
        # A float below 1 means a share of all training steps.
        warmup_steps=config.WARMUP_RATIO,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_steps=config.EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=config.LOGGING_STEPS,
        # Without this, a LoRA model may report no eval_loss.
        label_names=["labels"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=use_fp16,
        bf16=False,
        report_to=report_to,
    )
    # Pads each batch; label padding uses -100 so padding never counts toward the loss.
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX, padding=True)
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
        processing_class=tokenizer,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=float, default=config.EPOCHS)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop early, for a smoke test")
    parser.add_argument("--push", action="store_true", help="Upload the adapter to the Hub")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.environ.setdefault("WANDB_PROJECT", config.WANDB_PROJECT)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    data = load_dataset(config.DATASET_REPO)
    train_data = tokenize_split(data["train"], tokenizer)
    eval_data = tokenize_split(data["validation"], tokenizer)

    model = load_quantized_model()
    trainer = build_trainer(model, tokenizer, train_data, eval_data, args, "wandb", use_fp16=True)
    result = trainer.train()
    metrics = trainer.evaluate()
    console.print(f"Final train loss: {result.training_loss:.4f}")
    console.print(f"Final eval loss: {metrics['eval_loss']:.4f}")

    trainer.save_model(str(config.ADAPTER_DIR))
    tokenizer.save_pretrained(str(config.ADAPTER_DIR))
    if args.push:
        token = os.environ["HF_TOKEN"]
        trainer.model.push_to_hub(config.ADAPTER_REPO, token=token)
        tokenizer.push_to_hub(config.ADAPTER_REPO, token=token)
        console.print(f"Pushed adapter to {config.ADAPTER_REPO}")


if __name__ == "__main__":
    main()
