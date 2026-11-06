# CommitCraft guide

CommitCraft reads the changes you are about to commit and writes the commit message for you. It runs on
your own computer, works offline after setup, and sends your code nowhere.

No Python or AI knowledge is needed. You only need to know how to open a terminal.

## 1. Install three tools

You need about 3 GB of free disk space. Skip anything you already have.

| Tool | What it does | Get it |
| --- | --- | --- |
| Git | Tracks your code | https://git-scm.com/downloads |
| Ollama | Runs the AI model | https://ollama.com/download |
| uv | Installs CommitCraft | Run the command below |

**Windows** (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Mac or Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close the terminal and open a new one after installing.

## 2. Install CommitCraft

Run these once. The second command downloads a model of about 1 GB, so it can take a few minutes.

```bash
uv tool install git+https://github.com/shivankarpatra04/CommitCraft
commitcraft setup
```

You are done when it prints `Done`.

## 3. Use it

Go to any project that uses Git. Run these whenever you want to commit:

```bash
git add .
commitcraft commit
```

`git add .` selects all your changes. To commit only some files, name them instead, like `git add main.py`.

CommitCraft shows a message and asks what to do:

```text
fix(calc): raise error when inputs are None

[a]ccept, [e]dit, [r]etry, [q]uit [a]:
```

| Type | What happens |
| --- | --- |
| `a` | Commits with this message. Pressing Enter does the same. |
| `e` | Lets you change the message. |
| `r` | Writes a new message. |
| `q` | Stops without committing. Your changes stay selected. |

**Read the message before you accept it.** The model can get details wrong.

The first run after starting your computer is slower, because the model has to load. It took about
a minute on our test PC, then 2 to 4 seconds per message.

## Optional: use plain `git commit`

Run this once inside a project:

```bash
commitcraft install-hook
```

Now `git commit` opens your editor with the message already filled in. It stays out of the way for
`git commit -m`, merges and amends, and your commit still works if the model is unavailable.

## If something goes wrong

| You see | What to do |
| --- | --- |
| `commitcraft` or `uv` is not recognized | Close the terminal and open a new one. If it still fails, run `uv tool update-shell`, then open a new one again. |
| `Ollama is not installed` | Install Ollama from step 1, and open a new terminal. |
| `Ollama is not running` | Open the Ollama app, or run `ollama serve` in another terminal. |
| `Model commitcraft not found` | Run `commitcraft setup`. |
| `No staged changes` | Run `git add .` first. |
| `not a git repository` | Go to your project folder. To start a new project there, run `git init`. |
| `Author identity unknown` | Run `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"`. |
| The download stopped | Run `commitcraft setup` again. It continues from where it stopped. |
| `Warning: You are sending unauthenticated requests` | Ignore it. No account is needed. |

## What it does not do well

- It works best on small changes. It reads only the first 2,400 characters of a change, so commit big changes in smaller parts.
- It writes one line, with no longer description. Use `e` if you want to change it.
- Non-commercial use only, because the training data has a non-commercial license.

## Remove it

```bash
ollama rm commitcraft
uv tool uninstall commitcraft
```

If you ran `install-hook`, also delete `.git/hooks/prepare-commit-msg` in that project.
