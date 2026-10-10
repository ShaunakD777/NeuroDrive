# NeuroDrive

An interactive laboratory for understanding how neural networks learn, built around a 2D racing
simulator with three linked views: **Environment**, **Neural Network** and **Experiment**.

> How do different learning algorithms turn the same inputs and driving environment into better
> decisions, and what can their successes and failures teach us about AI?

## Status
Planning complete. Implementation starts at task 0.1 (see `docs/CHECKLIST.md`).

## Getting started
1. Create an **empty** GitHub repository (no auto-generated README, .gitignore or licence).
2. Clone it, copy these planning files into it, and make the first commit on `main`:
   `docs: planning documents`. Push.
3. In the repository's **Settings → Code security**, turn on secret scanning and push protection
   where available.
4. Open the folder in Claude Code. It reads `CLAUDE.md` automatically.
5. Run `/do-task 0.1`. Claude Code creates the `phase-0-foundations` branch and starts there.

## Working with Claude Code
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
- From task 0.6: pre-commit hooks (secret scan, private-key detection, large-file check) run before
  every commit, and GitHub Actions repeats the checks on every push.

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
