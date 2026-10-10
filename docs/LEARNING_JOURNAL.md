# NeuroDrive — Learning Journal

One entry per verified task (template in `WORKFLOW.md` §5). Newest at the bottom.
Claude Code drafts each entry; the user may edit or add notes in their own words.

---

### 2026-10-10 — Planning
- Concept: one environment, three feedback channels (teacher labels, episode summary, step reward).
- Key idea: loss, fitness and reward are different objectives; only the Driving Exam is compared
  across methods.
- Open question: which method will generalise best to held-out tracks, and why?

### 2026-10-10 — Task 0.1 Python project skeleton
- Concept: a package (importable folder), a virtual environment (private Python + libraries in
  `.venv/`), a lock file (exact versions of everything installed), a test runner, and an entry
  point that turns a function into the `nd` terminal command.
- Key equation(s) in words: none, plumbing task.
- What we observed: `uv sync` downloaded its own Python 3.12 even though the system has 3.11, and
  wrote `uv.lock`; pytest found and passed 1 test; `nd --help` lists the `version` subcommand.
- Surprise or difficulty: a Typer app with a single command collapses into that command, so a
  callback is needed to keep `nd` a group of subcommands. Ruff 0.17 also checks Python code
  blocks inside Markdown files.
- Open question: `pyproject.toml` says what we want, `uv.lock` says what we got — how strict
  should we be about upgrading the lock, given that results must stay reproducible?
