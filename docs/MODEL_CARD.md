---
license: cc-by-nc-4.0
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
datasets:
  - Shivankar04/commitcraft-data
tags:
  - lora
  - qlora
  - gguf
  - commit-messages
  - conventional-commits
---

# CommitCraft: Qwen2.5-Coder 1.5B for commit messages

This model turns a `git diff` into a one-line [Conventional Commit](https://www.conventionalcommits.org/)
message. It runs offline with Ollama. This repo holds the LoRA adapter and a 986 MB Q4_K_M GGUF.

- Code and setup guide: https://github.com/shivankarpatra04/CommitCraft

## Intended use

Suggesting a commit message for a small staged change, for a person to read and edit before committing.
It is not meant for unattended commits, large refactors or code review. Non-commercial use only, see
[License](#license).

## Use with Ollama

With [Ollama](https://ollama.com/download) and [uv](https://docs.astral.sh/uv/) installed:

```bash
uv tool install git+https://github.com/shivankarpatra04/CommitCraft
commitcraft setup
git add .
commitcraft commit
```

To call the model yourself, the prompt must match training:

- System: `You write Conventional Commit messages.`
- User: `Write a commit message for this diff:` followed by a new line and the diff

Real diffs should have their index hashes written as `index <HASH>..<HASH>`, as in the training data.
The CLI does this for you.

## Data

[`Shivankar04/commitcraft-data`](https://huggingface.co/datasets/Shivankar04/commitcraft-data): 4,201 rows
(train 3,767, validation 209, test 225), split by project, so no project is in two splits. The diffs come
from [CommitBench](https://huggingface.co/datasets/Maxscha/commitbench) (CC BY-NC 4.0). Claude Haiku
rewrote each message from the diff, and rows with low confidence or a bad format were dropped.

## Training

| Setting | Value |
| --- | --- |
| Quantization | 4-bit nf4, double quant, fp16 compute |
| LoRA | r=16, alpha=32, dropout=0.05, all linear layers except the output head |
| Trainable weights | 18.5M of 1.56B (1.18%) |
| Epochs | 2 (472 steps) |
| Learning rate | 2e-4, cosine, 3% warmup |
| Batch | 4, with 4 gradient accumulation steps |
| Loss | Commit message tokens only |
| Hardware | One Kaggle T4, 48 minutes |

Final eval loss was 1.2164. Eval loss was nearly flat after epoch 1, while train loss kept falling to about
0.76, so the second epoch shows some overfitting.

## Results

200 unseen test rows, greedy, at most 40 new tokens, with the 4-bit base model and adapter on a T4. A is the
base model and B is this model. Claude Haiku judged each pair without knowing the source, in a random order.

| Metric | A: base | B: this model |
| --- | --- | --- |
| Format compliance | 0.0% | 95.5% |
| Correct type | 0.0% | 79.0% |
| ROUGE-L | 0.174 | 0.492 |
| Judge score (1-5) | 1.59 | 4.16 |
| Average latency (GPU) | 2.88 s | 1.60 s |

B won 195 of 199 judged pairs, A won 1, and 3 were ties.

Read these numbers with care:

- The format and type checks read the raw output. The base model opens with a markdown fence, so it fails
  them even when a good message follows. The prompt gives no format rules either.
- Claude wrote the labels and also judged, so the judge may favour that style. ROUGE-L is also measured
  against Claude's labels.
- The 40-token limit cuts off the base model's long replies. B's lower latency comes from stopping earlier.
- The results above are not from the GGUF. On a 20-row check through Ollama, the GGUF passed format on 20 of
  20 and matched the label's type on 13 of 20. That is too few rows for a firm comparison.

## Limitations

- It was trained and tested only on small diffs: at most 4,000 characters and 3 files. The longest training
  diff is 469 tokens. Expect weaker results on big changes. Only the first 2,400 characters are read.
- Rare types are weak. `fix` is about half of the labels, and `ci` has only 1 in the labeled data.
- It can state wrong details, or pick a different type or scope from what a person would choose.
- It writes only a subject line, with no body.
- Training diffs hide numbers as `<I>`, but real diffs have real numbers, so input differs slightly.
- Training covers mainly Python, JavaScript, PHP, Ruby, Java and Go.

## License

The training data is CC BY-NC 4.0, so this adapter and the GGUF are for non-commercial use, with credit to
CommitBench. The base model is Apache-2.0.
