"""Gradio demo: paste a git diff and get a commit message, on CPU with llama.cpp."""

import time
from functools import lru_cache

import config

EXAMPLE_DIFFS = [
    """diff --git a/src/auth/session.py b/src/auth/session.py
index 3f2a91c..8bd40e7 100644
--- a/src/auth/session.py
+++ b/src/auth/session.py
@@ -12,6 +12,8 @@ def load_session(token):
     data = cache.get(token)
+    if data is None:
+        return None
     return Session(user_id=data["user_id"], expires=data["expires"])
""",
    """diff --git a/web/components/SearchBox.js b/web/components/SearchBox.js
index a71c04e..d29f3b1 100644
--- a/web/components/SearchBox.js
+++ b/web/components/SearchBox.js
@@ -4,10 +4,17 @@ export function SearchBox({ onSearch }) {
   const [query, setQuery] = useState("");
+  const [recent, setRecent] = useState(loadRecentSearches());
+
+  function submit() {
+    saveRecentSearch(query);
+    setRecent(loadRecentSearches());
+    onSearch(query);
+  }
+
   return (
-    <form onSubmit={() => onSearch(query)}>
+    <form onSubmit={submit}>
       <input value={query} onChange={(e) => setQuery(e.target.value)} />
+      <RecentList items={recent} onSelect={setQuery} />
     </form>
   );
""",
    """diff --git a/README.md b/README.md
index 09c7de2..f41ab6d 100644
--- a/README.md
+++ b/README.md
@@ -18,3 +18,9 @@ Run the server with `make run`.
 ## Tests

 Run `make test`.
+
+## Configuration
+
+Set `DATABASE_URL` before starting the server. The default is a local
+SQLite file named `dev.db`.
""",
]


@lru_cache(maxsize=1)
def load_model():
    """Load the GGUF once. Use a local copy if there is one, otherwise download it from the Hub."""
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    local = config.ROOT_DIR / config.GGUF_FILENAME
    path = local if local.exists() else hf_hub_download(config.ADAPTER_REPO, config.GGUF_FILENAME)
    return Llama(
        model_path=str(path),
        n_ctx=config.DEMO_CONTEXT_TOKENS,
        chat_format="chatml",  # The same format the model was trained on.
        verbose=False,
    )


def generate(diff: str, llm=None) -> tuple[str, str]:
    """Return the commit message and a short status line."""
    if not diff.strip():
        return "", "Paste a diff first."
    llm = llm or load_model()
    start = time.perf_counter()
    reply = llm.create_chat_completion(
        messages=config.build_messages(config.normalize_diff(diff)),
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_NEW_TOKENS,
        repeat_penalty=1.0,  # The evaluated model used no repetition penalty.
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    seconds = time.perf_counter() - start
    lines = reply["choices"][0]["message"]["content"].strip().splitlines()
    status = f"Generated in {seconds:.1f} s."
    if len(diff) > config.MAX_PROMPT_DIFF_CHARS:
        status += f" Only the first {config.MAX_PROMPT_DIFF_CHARS:,} characters were read."
    return (lines[0].strip() if lines else ""), status


def build_demo():
    """Build the page: one textbox, three examples, the message and the generation time."""
    import gradio as gr

    with gr.Blocks(title="CommitCraft") as demo:
        gr.Markdown(
            "# CommitCraft\n"
            "Turns a `git diff` into a Conventional Commit message, using a fine-tuned "
            "Qwen2.5-Coder 1.5B on CPU. It was trained on small diffs (up to 4,000 characters), "
            "and it can get details wrong, so read the message before you use it. "
            "The first request also loads the model, so it is slower. "
            f"[GitHub]({config.GITHUB_URL})"
        )
        diff = gr.Textbox(label="git diff", lines=14, placeholder="Paste the output of git diff")
        button = gr.Button("Write commit message", variant="primary")
        message = gr.Textbox(label="Commit message", interactive=False)
        status = gr.Markdown()
        gr.Examples(examples=[[example] for example in EXAMPLE_DIFFS], inputs=diff)
        button.click(generate, inputs=diff, outputs=[message, status], api_name="generate")
    return demo


if __name__ == "__main__":
    build_demo().launch()
