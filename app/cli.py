"""CommitCraft CLI: write a commit message for staged changes with a local Ollama model."""

import subprocess
from pathlib import Path

import requests
import typer

import config

app = typer.Typer(help=__doc__, no_args_is_help=True, add_completion=False)

HOOK_MARKER = "# Installed by commitcraft install-hook."
HOOK = f"""#!/bin/sh
{HOOK_MARKER}
# Git sets $2 for -m, templates, merges, squashes and amends. Only fill a blank message.
[ -z "$2" ] || exit 0
# A failed suggestion must never block the commit.
message=$(commitcraft suggest 2>/dev/null) || exit 0
[ -n "$message" ] || exit 0
{{ printf '%s\\n' "$message"; cat "$1"; }} > "$1.commitcraft" && mv "$1.commitcraft" "$1"
"""


def fail(message: str) -> typer.Exit:
    """Print a clear error. Callers raise the result, so it is clear that the command stops."""
    typer.secho(message, err=True, fg=typer.colors.RED)
    return typer.Exit(1)


def run_git(*args: str) -> str:
    """Run git and return its output, or stop with git's own error."""
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except FileNotFoundError:
        raise fail("Git is not installed.") from None
    if result.returncode != 0:
        raise fail(result.stderr.strip() or f"git {args[0]} failed.")
    return result.stdout


def staged_diff() -> str:
    """Read staged changes in the same diff format as the training data."""
    run_git("rev-parse", "--git-dir")  # Gives a clear error outside a repo.
    diff = run_git(
        "diff", "--cached", "--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/"
    )
    if not diff.strip():
        raise fail("No staged changes. Stage files with `git add` first.")
    return config.normalize_diff(diff)


def ask_model(diff: str) -> str:
    """Ask the local Ollama model for a message. build_messages also cuts long diffs."""
    body = {
        "model": config.OLLAMA_MODEL,
        "messages": config.build_messages(diff),
        "stream": False,
        "options": {"temperature": config.TEMPERATURE, "num_predict": config.MAX_NEW_TOKENS},
    }
    url = f"{config.OLLAMA_URL}/api/chat"
    try:
        response = requests.post(url, json=body, timeout=config.OLLAMA_TIMEOUT_SECONDS)
    except requests.ConnectionError:
        raise fail("Ollama is not running. Start it with `ollama serve`.") from None
    except requests.Timeout:
        raise fail(f"Ollama did not reply in {config.OLLAMA_TIMEOUT_SECONDS} s.") from None
    if response.status_code == 404:
        raise fail(
            f"Model `{config.OLLAMA_MODEL}` not found. "
            f"Run `ollama create {config.OLLAMA_MODEL} -f Modelfile`."
        )
    if not response.ok:
        raise fail(f"Ollama returned HTTP {response.status_code}.")
    lines = response.json()["message"]["content"].strip().splitlines()
    if not lines:
        raise fail("The model returned an empty message.")
    return lines[0].strip()


@app.command()
def suggest() -> None:
    """Print a commit message for the staged changes."""
    typer.echo(ask_model(staged_diff()))


@app.command()
def commit() -> None:
    """Suggest a message, then accept, edit or retry it before committing."""
    diff = staged_diff()
    message = ask_model(diff)
    while True:
        typer.echo(f"\n{message}\n")
        choice = typer.prompt("[a]ccept, [e]dit, [r]etry, [q]uit", default="a").strip().lower()
        if choice.startswith("a"):
            break
        if choice.startswith("e"):
            message = typer.prompt("Message", default=message).strip()
        elif choice.startswith("r"):
            message = ask_model(diff)
        elif choice.startswith("q"):
            typer.echo("Nothing committed.")
            return
    typer.echo(run_git("commit", "-m", message).strip())


@app.command("install-hook")
def install_hook() -> None:
    """Fill the commit message box automatically on a plain `git commit`."""
    hook = Path(run_git("rev-parse", "--git-path", "hooks").strip()) / "prepare-commit-msg"
    if hook.exists() and HOOK_MARKER not in hook.read_text(encoding="utf-8", errors="replace"):
        raise fail(f"{hook} already exists and is not ours. Remove it first.")
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(HOOK, encoding="utf-8", newline="\n")
    hook.chmod(0o755)
    typer.echo(f"Installed {hook}")


if __name__ == "__main__":
    app()
