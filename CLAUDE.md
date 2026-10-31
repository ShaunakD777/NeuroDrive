# NeuroDrive — Instructions for Claude Code

NeuroDrive is a learn-by-building platform: a 2D racing simulator whose cars are driven by neural
networks, plus three linked views (Environment, Neural Network, Experiment) and a set of learning
algorithms implemented from first principles.

**Central question:** *How do different learning algorithms turn the same inputs and driving
environment into better decisions, and what can their successes and failures teach us about AI?*

Read `docs/PROJECT_CONCEPT.md` once per session if you have not seen it. It explains the *why*.

---

## 1. Who you are working with

The user is a final-year Computer Science (AIML) student learning neural networks from first
principles. **You write the code; the user learns from your explanations.** The explanation is
as much a deliverable as the code.

Explanation style (mandatory):
- Plain language, concrete and procedural: "first it does X, then Y, because Z".
- **Do not paste code blocks into explanations.** Refer to files and functions by name and
  describe what they do in words. Quote at most one short line when it is truly unavoidable.
- Mathematics: define every symbol in words, then give a small worked numerical example
  (2–3 inputs, round numbers) and show that the code produces the same number.
- Algorithms: give numbered plain-language pseudocode steps, not code.
- Define ML terms on first use (see `docs/GLOSSARY.md`; add new terms there).

## 2. Map of the docs

| File | Use it for |
|---|---|
| `docs/PROJECT_CONCEPT.md` | Vision, central question, the three views, the three feedback channels |
| `docs/ARCHITECTURE.md` | Stack, module layout, **contracts** (observation, action, network file, run folder, API) |
| `docs/ROADMAP.md` | Every phase and task, in order, with files, concept and verification |
| `docs/CHECKLIST.md` | Status of every task — the single source of truth for progress |
| `docs/WORKFLOW.md` | The per-task loop and the Brief / Report / Experiment templates |
| `docs/EXPERIMENT_PROTOCOL.md` | The Driving Exam, fair-comparison rules, statistics, planned studies |
| `docs/DECISIONS.md` | Architecture Decision Records (ADRs). Add one before any new architectural choice |
| `docs/ARCHITECTURE.md` §11 | What may and may not be committed; branches, commits and versions |
| `docs/LEARNING_JOURNAL.md` | One entry per completed task: what was learned, what was observed |
| `task-notes/` | Every task's Brief and Report, saved as files (local only, git-ignored; ADR-029) |
| `docs/GLOSSARY.md` | Plain-language definitions |

## 3. Working rules (non-negotiable)

1. **One task at a time.** Tasks have IDs like `2.7` in `docs/ROADMAP.md`. Work only on the task
   the user names. **Never start the next task automatically.**
2. **Follow the task loop** in `docs/WORKFLOW.md`: Brief → Implement → Explain → Verify → Report →
   Stop. If the user asks for "brief only", stop after the Brief. Every Brief and Report is also
   saved to `task-notes/` (`docs/WORKFLOW.md` §2a); that folder is never committed.
3. **Stay in scope.** Touch only the files the task lists. If another file must change, say so in
   the Brief with the reason. No unrelated refactors, no silent rewrites of working code.
4. **Never claim something works without running its verification.** Show the command you ran and
   summarise the real output. If it fails, say so plainly.
5. **Checklist discipline.** In `docs/CHECKLIST.md`: mark ◐ when you start, ☑ when implemented,
   ✅ only after verification passed. Never mark ✅ on an unverified task.
6. **Decisions.** If a task needs an architectural choice not covered by `docs/ARCHITECTURE.md` or
   `docs/DECISIONS.md`, stop, present the options with trade-offs and a recommendation, and wait.
   After the user decides, record an ADR.
7. **Journal.** After each verified task, append an entry to `docs/LEARNING_JOURNAL.md`.
8. **Commit** after each verified task on the current phase branch: message
   `task X.Y: <short summary>`. Follow the git rules in §5. Push only as part of
   `/finish-phase`, or when the user asks.
9. **Questions beat progress.** If the user asks a question mid-task, answer it fully first.
10. **Escalate** (stop and report) when: verification fails after two fix attempts; the task turns
    out larger than described (propose a split); or a contract would need to change.

## 4. Architecture rules (non-negotiable)

- **Python is the single source of truth.** All simulation, physics, network maths, learning and
  explanation maths live in Python. The web app only *displays* data and *sends commands*. It
  never simulates or computes network outputs itself.
- **Layering** (arrows = "may import"):
  `core` ← `env` (uses only `core/rng`, except `env/policies.py`, which adapts `core/nn`
  networks into policies) ← `learn` ← `experiment` ← `server`; `explain` may import `core` and `env`.
  Full rules: `docs/ARCHITECTURE.md` §5. Nothing under `src/neurodrive` except `server/` imports
  FastAPI. Nothing imports `server/` except the CLI entry point.
- **From scratch:** no PyTorch, TensorFlow, JAX, scikit-learn or autograd libraries anywhere in
  `src/neurodrive`. **NumPy is allowed** for array arithmetic. `reference/` may use PyTorch *only*
  to cross-check our results, and is never imported by `src/`.
- **Randomness:** every random number comes from an explicit, seeded generator object passed in as
  an argument (see `src/neurodrive/core/rng.py`). Never use `np.random.<function>` globals or the
  `random` module.
- **Determinism:** same config + same seed + same code ⇒ identical results. Tests enforce this.
- **Contracts** in `docs/ARCHITECTURE.md` §6 (observation, action, track, network file, episode
  recording, learner, run folder, exam result) change only through an ADR plus a version bump.
- **Units:** metres, seconds, radians internally. Degrees only in the UI.
- **Batching:** the environment and population forward pass operate on arrays of N cars at once;
  a single car is just N = 1.
- Type hints on all public functions. Docstrings describe the maths in words with symbols defined.

## 5. Git, GitHub and secrets (non-negotiable)

The repository is public-safe at all times (ADR-014, `docs/ARCHITECTURE.md` §11).

- **Branches:** each phase is developed on `phase-<N>-<short-name>`. Never commit directly to `main`
  during a phase. At the start of a phase, create its branch from an up-to-date `main`.
- **Commits:** one commit per verified task. Before committing, run `git status`, check every file
  in the list, and stage files **by name**. Never use `git add -A` or `git add .`. Pre-commit hooks
  must pass; never bypass them with `--no-verify`.
- **Versions:** a completed phase is merged into `main` and tagged `phase-<N>` using `/finish-phase`.
  Ask the user before every push.
- **Never:** force-push, rewrite pushed history, delete tags, or commit `runs/`, `data/`,
  databases, recordings, `node_modules/` or build output.
- **Secrets:** NeuroDrive needs none. Never write a token, password, API key, private key or
  personal data (e-mail, absolute paths with user names) into code, configs, docs, tests, logs or
  run folders. Never read real `.env` files (`.env`, `.env.local`, `.env.production`, ...); the
  placeholder files `.env.example` and `.env.local.example` may be read and edited, and must
  never hold a real value. If a secret is ever needed, read it from an `ND_`-prefixed
  environment variable and add only a placeholder to `.env.example`, after an ADR.
- **If a secret is found in a commit:** stop, tell the user, do not push. A pushed secret must be
  revoked first; history cleaning comes second.
- GitHub sign-in is handled by the user's own git credentials. Never ask for, store or print a token.

## 6. Commands

These are created in Phase 0. If a task changes a command, update this section in that task.

**Prerequisites (once per machine):** `uv` (it fetches Python 3.12 by itself), Node.js with npm
(tested with Node 24), and, for the web smoke test, `cd web && npx playwright install chromium`.

| Purpose | Command |
|---|---|
| Install Python deps | `uv sync` |
| All Python tests | `uv run pytest` |
| One area, quiet | `uv run pytest tests/core -q` |
| Lint and format | `uv run ruff check . && uv run ruff format .` |
| Start backend (port 8000) | `uv run python -m neurodrive serve` (or `uv run nd serve`; add `--runs-dir <folder>` to show other runs) |
| Start web app (port 5173) | `cd web && npm install && npm run dev` |
| Regenerate the track files in `tracks/` | `uv run python scripts/make_tracks.py` |
| Train from a config into `runs/` (toy configs: `configs/toy_line.yaml`, `toy_sine.yaml`, `toy_xor.yaml`, `toy_two_spirals.yaml`; options `--seed`, `--tag`, `--epochs`, `--runs-dir`) | `uv run python -m neurodrive train --config configs/toy_xor.yaml` (or `uv run nd train ...`) |
| Run several seeds (one run folder per seed, tag `<tag>-s<seed>`, shared `sweep` id; prints median and IQR of the final loss) | `uv run python -m neurodrive sweep --config configs/toy_sine.yaml --seeds 0-4` (or `uv run nd sweep ...`) |
| Driving Exam on a checkpoint | `uv run nd exam --run runs/<run_id> --unit <n>` |
| Rebuild / check the SQLite mirror | `uv run nd db rebuild` / `uv run nd db check` |
| Web unit tests / smoke test | `cd web && npm run test` / `npm run e2e` |
| Regenerate TypeScript types from the Python message models | `cd web && npm run gen:types` |
| PyTorch cross-check of our gradients and optimizers (PyTorch fetched for that run only; `reference/` is never imported by `src/`) | `uv run --with torch python reference/torch_check.py` (its tests: `uv run --with torch python -m pytest tests/reference`) |
| Run all pre-commit hooks | `uv run python -m pre_commit run --all-files` |
| Install the git hooks (once per clone) | `uv run python -m pre_commit install` |

The pre-commit commands use `python -m pre_commit` because Windows Smart App Control can block the
generated `pre-commit.exe` launcher; the installed git hook already calls Python this way. For the
same reason the backend is started with `python -m neurodrive` (the `nd.exe` launcher is blocked on
the development machine; the browser tests start it this way too), and `uv run python -m pytest`
works wherever `uv run pytest` does not. If Smart App Control blocks Python's own `asyncio` DLLs in
the Python that uv downloads (seen on 2026-10-25), install Python 3.12 from python.org and run
`uv sync --python <path to its python.exe>`; the browser tests also train two small toy runs into
`runs/` (tag `e2e`) the first time.

**Not available yet** (the command is listed so its final form is fixed; running it today gives
"No such command"): `nd exam` arrives in task 5.4 and `nd db` in 5.5b.

## 7. Project slash commands

- `/brief <task-id>` — produce the task Brief only, no file changes.
- `/do-task <task-id>` — full loop for one task, then stop.
- `/explain <file, function or concept>` — plain-language explanation of existing code or a concept.
- `/experiment <study-id or question>` — run an experiment by the protocol and write the report.
- `/status` — where we are, what is verified, what is next.
- `/finish-phase <N>` — check a phase is complete, merge it into `main`, tag `phase-<N>`, push after confirmation.

## 8. Where we are

See `docs/CHECKLIST.md` for the current task and every task's status. Each phase is built on its own
branch `phase-<N>-<short-name>` (the first is `phase-0-foundations`), as described in §5.
