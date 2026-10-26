# NeuroDrive

An interactive laboratory for understanding how neural networks learn, built around a 2D racing
simulator with three linked views: **Environment**, **Neural Network** and **Experiment**.

> How do different learning algorithms turn the same inputs and driving environment into better
> decisions, and what can their successes and failures teach us about AI?

## Status
Phase 3 ("Backpropagation (toy lab)") complete, tagged `phase-3`: loss functions and activation
derivatives, backpropagation for one neuron and for the whole multilayer network (batched), a
gradient checker that agrees with finite differences on every activation and architecture,
sensitivity computed by backprop, toy tasks (line, sine, XOR, two spirals) trained by plain gradient
descent into run folders, the toy lab in the Experiment view (loss curve, predictions, edge glow by
gradient size), and study S2b on activations and vanishing gradients (`docs/studies/`). Phase 2
("Multilayer network and Network view") is tagged `phase-2`, Phase 1 ("One neuron can drive")
`phase-1` and Phase 0 (foundations) `phase-0`. Next: Phase 4, "Optimisers and experiment basics"
(see `docs/CHECKLIST.md`).

## Prerequisites
- [uv](https://docs.astral.sh/uv/) for Python. It downloads Python 3.12 by itself; your system
  Python is not touched.
- [Node.js](https://nodejs.org/) with npm (tested with Node 24) for the web viewer.
- Git.

## Setup (once per clone)
1. `uv sync`: create `.venv/` and install the Python dependencies exactly as pinned in `uv.lock`.
2. `uv run python -m pre_commit install`: install the git hooks that check every commit.
3. `cd web && npm ci`: install the web dependencies exactly as pinned in `package-lock.json`.
4. `cd web && npx playwright install chromium`: download the browser used by the smoke test
   (only needed for `npm run e2e`).

## Everyday commands
| What | Command |
|---|---|
| Python tests | `uv run pytest` |
| Lint and format | `uv run ruff check . && uv run ruff format .` |
| Start the server (port 8000) | `uv run nd serve` |
| Start the web viewer (port 5173) | `cd web && npm run dev`, then open http://127.0.0.1:5173 |
| Web unit tests / browser smoke test | `cd web && npm run test` / `npm run e2e` |
| Regenerate TypeScript types after changing `server/schemas.py` | `cd web && npm run gen:types` |
| Run every commit check on all files | `uv run python -m pre_commit run --all-files` |

The full list, including commands that later phases add, is in `CLAUDE.md` §6.

## Contribution rules
- Work happens on a phase branch (`phase-<N>-<short-name>`), never directly on `main`.
- One commit per verified task, with the message `task X.Y: <summary>`; documentation-only commits
  use `docs: <summary>`.
- Run `git status` first and stage files **by name**. Never `git add -A` or `git add .`.
- Never bypass a failing hook with `--no-verify`: fix the cause and commit again.
- Never force-push, rewrite pushed history, or move or delete a tag.
- Every random number comes from an explicitly seeded generator (`neurodrive.core.rng`); no ML
  frameworks inside `src/neurodrive` (NumPy only).

## Working with Claude Code
Open the folder in Claude Code; it reads `CLAUDE.md` automatically.
- `/status`: where we are
- `/brief 0.1`: explanation and plan for a task, without changes
- `/do-task 0.1`: build, verify, explain and commit one task, then stop
- `/explain <file or concept>`: plain-language explanation
- `/experiment S4`: run a study by the protocol
- `/finish-phase 0`: merge a completed phase into `main`, tag it, push after confirmation

## Versions
- Each phase is built on its own branch (`phase-<N>-<short-name>`), one commit per verified task.
- A completed phase is merged into `main` and tagged **`phase-<N>`**. Each tag is a saved,
  working version of the project at the end of that stage.
- `main` always contains the last completed phase.

## Keeping the repository safe to publish
- NeuroDrive needs no secrets. If one is ever needed, it lives in a git-ignored `.env`; only
  placeholders go in `.env.example`.
- `.gitignore` keeps out run outputs, databases, recordings, dependencies, keys and local settings.
- Pre-commit hooks run before every commit: detect-secrets (secret scan), private-key detection,
  a 1000 KB large-file check, whitespace fixers, YAML syntax, and ruff. A commit that fails a check
  is refused. Known non-secrets would be listed in `.secrets.baseline` (there are none).
- GitHub Actions (`.github/workflows/ci.yml`) runs on every push: Python lint and tests, the web
  build and unit tests, a check that the generated TypeScript types are up to date, and a gitleaks
  scan of the full history.
- In the repository's **Settings → Code security**, secret scanning and push protection should be
  on (where GitHub offers them).
- **Windows note:** if Smart App Control is on, it may block compiled programs such as the
  `pre-commit.exe` launcher or a hook's own program. Use `uv run python -m pre_commit ...` as shown
  above; the installed git hook already works this way, and the Python-based hooks are started
  through Python for the same reason (ADR-021).

## Documents
- `docs/PROJECT_CONCEPT.md`: vision and the three feedback channels
- `docs/ARCHITECTURE.md`: stack, layout, contracts, API, repository hygiene (§11)
- `docs/ROADMAP.md`: all phases and tasks
- `docs/CHECKLIST.md`: progress
- `docs/EXPERIMENT_PROTOCOL.md`: the Driving Exam and fair-comparison rules
- `docs/WORKFLOW.md`: the per-task loop, templates and version-control steps
- `docs/DECISIONS.md`: architecture decisions
- `docs/LEARNING_JOURNAL.md`, `docs/GLOSSARY.md`

## Stack (summary)
Python 3.12 core (NumPy only, algorithms from scratch) · FastAPI server · React + TypeScript viewer
· SQLite mirror for querying runs · GitHub Actions CI.
