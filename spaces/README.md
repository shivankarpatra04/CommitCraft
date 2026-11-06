---
title: CommitCraft
emoji: 📝
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
license: cc-by-nc-4.0
models:
  - Shivankar04/commitcraft-qwen-1.5b-lora
datasets:
  - Shivankar04/commitcraft-data
---

# CommitCraft

Paste a `git diff` and get a Conventional Commit message. The model is Qwen2.5-Coder 1.5B fine-tuned
with QLoRA, running on CPU with llama.cpp.

- Code: https://github.com/shivankarpatra04/CommitCraft
- Model, results and limitations: https://huggingface.co/Shivankar04/commitcraft-qwen-1.5b-lora

The training data is CC BY-NC 4.0, so this model is for non-commercial use.
