# Interview notes

Questions and answers from each phase of CommitCraft.

## Phase 1: Scaffold

Q1. Why keep the prompt template in one place?
A. The model must see the same prompt in training and in real use. If the wording differs
even a little, quality can drop without any error. One definition in `config.py` prevents that.

Q2. Why pin Python to 3.11 and commit `uv.lock`?
A. ML libraries like `bitsandbytes` and `torch` often lag behind new Python versions. The
lockfile records exact package versions, so a local run, CI and Kaggle setup stay consistent.

Q3. What does the commit regex accept, and why does it matter?
A. It needs an allowed type, an optional lowercase scope, a colon and space, then a lowercase
subject of 4 to 72 characters with no trailing period. It filters training data and also
measures format compliance in eval, so both use the same definition of "correct".

## Phase 2: Raw data

Q1. Why stream the dataset instead of downloading it?
A. CommitBench has over a million rows and we need only 20,000. Streaming reads rows one by
one and stops early, so it took about 19 seconds and used little disk.

Q2. What real-data problem did you find, and how?
A. CommitBench replaces numbers with `<I>`, so `Bump version to <I>` passed my version-bump
regex, which looked for digits. Hand-made tests passed because they used real digits. I found it
by reading actual output, then treated `<I>` as a number and added tests for it.

Q3. Why cap rows per language?
A. Without a cap the first 20,000 rows would mostly be Python and JavaScript, and the model
would do worse on other languages. The cap gives a more balanced set: py, js and php hit 4,000,
while rb, java and go got 2,268 to 3,398.

## Phase 3: Relabel and publish

Q1. Why split by project instead of randomly?
A. Commits from one repo share file names, style and vocabulary. With a random split the test
set would contain repos the model already saw, so scores would look better than real use. Splitting
by project tests the model on code it has never seen.

Q2. Why relabel with Claude instead of using the original messages?
A. Original messages have mixed styles and most do not follow Conventional Commits. Claude acts
as a teacher and rewrites them into one format. The diff is the truth and the original is only a
hint. The risk is that labels can be wrong, so a format check and quality score filter them.

Q3. How did you control cost?
A. I ran 50 rows first and used the measured tokens to estimate the full run at $0.38, then got
approval. The Batch API costs 50% less, and resume logic prevents paying twice. The actual cost
was $0.379.

## Phase 4: Fine-tune

Q1. What was the hardest bug in training, and how did you fix it?
A. Training crashed on step 1 with a BFloat16 error from the grad scaler. I read the TRL source
and found `SFTTrainer` casts LoRA weights to bf16 on every quantized model, even with `bf16=False`.
The T4 has no native bf16, so I kept fp16 and cast the adapter back to float32 after building the
trainer. I reproduced it on CPU by flagging a tiny model as 4-bit, then confirmed it on the GPU.

Q2. Why is train loss 0.76 at the end but eval loss 1.216? Is that a problem?
A. The gap shows some overfitting: in epoch 2 train loss dropped sharply while eval loss stayed
near 1.22. It did not get worse, so it is not severe, and the trainer loads the best checkpoint at
the end. Fewer epochs would probably give a similar result. Loss cannot settle this, so Phase 5
compares real generated messages.

Q3. Why LoRA instead of training the whole model?
A. LoRA trains small adapter matrices and freezes the rest: 18.5M of 1.56B weights here. With 4-bit
weights, the model fits on a free T4, and the adapter is only 74 MB. It also reduces overfitting
risk on a small dataset of 3,767 rows.

## Phase 5: Evaluate

Q1. Why randomise the A/B order for the judge?
A. LLM judges can favour a position, such as the first message. If B were always second, that bias
would look like model quality. A seeded random swap per row removes it and stays repeatable. The code
then maps "message 1" and "message 2" back to the right model, and a unit test covers that mapping.

Q2. Base scored 0% on format. Does that mean the fine-tune made it that much better?
A. No, that number overstates it. The base model opens with a markdown fence, so the strict check fails
every row, even when a good message is inside. The prompt also gives no format rules. The fair claim
is that fine-tuning gives a ready-to-use single line. I also report ROUGE-L and judge scores for this reason.

Q3. What are the weak points of this evaluation?
A. Claude wrote the labels and also judges, so it may favour that style. Test diffs are as small as training
diffs, so big real diffs are untested. The "best examples" are chosen, not typical. An independent human
review of a random sample and a test on large diffs would be the next step.

## Phase 6: GGUF, Ollama CLI and git hook

Q1. How did you make sure the Ollama model behaves like the evaluated one?
A. Differences between training and serving often hurt quality without any error. I compared the
Modelfile template with the tokenizer's real output, and a test checks it. Ollama can add a repetition
penalty, so I set it to 1 as in eval. The CLI also uses the same prompt builder and the same diff format.

Q2. What does the git hook do, and how do you stop it from causing problems?
A. `prepare-commit-msg` fills the message on a plain `git commit`. It does nothing when git passes a
source, which covers `-m`, merges, squashes and amends. Any failure exits 0, so a stopped Ollama never
blocks a commit. `install-hook` also refuses to overwrite a hook it did not write.

Q3. Why does quantization matter here, and what did it cost?
A. Q4_K_M shrinks the 3.1 GB f16 model to 986 MB and makes it practical on a laptop CPU. It can reduce
quality. On 20 rows, format stayed at 20/20, while type matched the label on 13/20, below the 79% before
quantizing. 20 rows are too few to settle that, so I would rerun the 200-row eval on the GGUF.

## Phase 7: Demo and model card

Q1. Why is there no live demo link?
A. When I pushed, Hugging Face returned HTTP 402, because Gradio Spaces on free CPU now need a paid plan.
I chose not to pay for one, so there is no hosted demo. The project's main goal is offline use, so a local demo is enough.
chose no hosted demo. The project's main goal is offline use, so a local demo and a GIF show it well.

Q2. Why is the model licensed CC BY-NC 4.0 when the base model is Apache-2.0?
A. The adapter was trained on CommitBench, which is CC BY-NC 4.0. Whether a trained model inherits a data
license is not settled, so I took the cautious route: non-commercial use, with credit, and said so clearly.
For commercial use I would retrain on data with a permissive license. I am not a lawyer.

Q3. What did you change to make this easy for a beginner?
A. I tested the real path and counted commands. Setup needed three steps, so I made one command that
downloads the model, runs `ollama create` and deletes the extra copy. Its error messages say what to do next.
I also moved heavy data libraries out of the install, since CLI users do not need them.

## Phase 8: Polish

Q1. How do you test code that needs a GPU, a paid API and Ollama?
A. Split the logic from the expensive calls. Unit tests mock those calls and check prompts, masking and error
handling on CPU in seconds, and hook tests use real git. Mocks cannot prove the real thing works, so I also ran
one-off real checks: a tiny model for the trainer, and the real GGUF through Ollama. Real GPU runs still found bugs.

Q2. What is in `.dockerignore`, and why does it matter?
A. Docker sends the build folder to the daemon, and `COPY` could then put files into the image. Ignoring `.env` keeps
secrets out of a shareable image. Ignoring `outputs`, `.git` and `*.gguf` keeps the build fast and the image small.
The image downloads the model at the first request, and a compose volume keeps it between runs.

Q3. If you had one more week, what would you do?
A. First, rerun the 200-row eval on the GGUF, because quantized quality was checked on only 20 rows. Then test on real
large diffs, since that is the biggest known gap, and have a person review a random sample, so the judge is not the
same model that wrote the labels. Better data for rare types like `ci` would come after that.
