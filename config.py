"""All CommitCraft settings. Secrets stay in .env and are read with os.environ."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR = Path(__file__).parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
RAW_PATH = OUTPUTS_DIR / "raw.jsonl"
LABELED_PATH = OUTPUTS_DIR / "labeled.jsonl"
ADAPTER_DIR = OUTPUTS_DIR / "adapter"
PREDICTIONS_PATH = OUTPUTS_DIR / "eval_predictions.jsonl"
RESULTS_PATH = OUTPUTS_DIR / "results.md"

# Models and repos
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CLAUDE_MODEL = "claude-haiku-5-5"
SOURCE_DATASET = "Maxscha/commitbench"
HF_USERNAME = os.environ.get("HF_USERNAME", "")
DATASET_REPO = f"{HF_USERNAME}/commitcraft-data"
ADAPTER_REPO = f"{HF_USERNAME}/commitcraft-qwen-1.5b-lora"
GGUF_FILENAME = "commitcraft-q4_k_m.gguf"
OLLAMA_MODEL = "commitcraft"
OLLAMA_URL = "http://localhost:11434"

# The one prompt template, used in data, training, eval and the app
SYSTEM_PROMPT = "You write Conventional Commit messages."
USER_PROMPT = "Write a commit message for this diff:\n{diff}"
ASSISTANT_FORMAT = "<type>(<scope>): <subject>"

# The one Conventional Commit check
COMMIT_REGEX = (
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore)"
    r"(\([a-z0-9_\-./]+\))?: [a-z].{2,70}[^.]$"
)

# Data filters
MIN_DIFF_CHARS = 200
MAX_DIFF_CHARS = 4000
MAX_FILES_TOUCHED = 3
MIN_MESSAGE_CHARS = 10
MAX_MESSAGE_CHARS = 100
MAX_ROWS_PER_LANGUAGE = 4000
MAX_KEPT_ROWS = 20000

# Relabeling and splits
MIN_QUALITY = 4
SPLIT_RATIOS = {"train": 0.90, "validation": 0.05, "test": 0.05}

# QLoRA training
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 2
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
EVAL_STEPS = 100

# Generation
MAX_NEW_TOKENS = 40
TEMPERATURE = 0.2
EVAL_MAX_ROWS = 200
