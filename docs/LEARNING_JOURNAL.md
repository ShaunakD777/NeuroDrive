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

### 2026-10-10 — Task 0.2 Seeded randomness module
- Concept: a pseudo-random generator follows a fixed recipe set by its seed; each sub-component
  gets its own named child stream so changing one never shifts another.
- Key equation(s) in words: child stream = f(master seed, family-tree position, name). It does
  not depend on how many numbers the parent has produced or on the order children are requested.
- What we observed: seed 42 gives 0.774, 0.439, 0.859 every time; seed 43 gives 0.652, 0.044,
  0.020; the child "init" of seed 42 gives 0.890, 0.832, 0.118 even after the parent has drawn
  1000 numbers. A planted `np.random.rand` in the source was caught by the scan test.
- Surprise or difficulty: asking twice for the same child name returns identical streams, by
  design; a component must keep its child, not re-ask. Also, one test I first wrote could not
  fail, so it was removed: a test must be able to fail to prove anything.
- Open question: when evolution runs many seeds in parallel processes, how will each worker
  get its generator without sharing state?

### 2026-10-11 — Task 0.3 Server skeleton
- Concept: REST is request and response; a WebSocket is a connection that stays open so the server
  can push messages (live driving needs this). A health endpoint answers "are you up, and which
  version?".
- Key equation(s) in words: none, plumbing task.
- What we observed: `nd serve` starts on port 8000; `/api/health` returns status ok and the
  version; a WebSocket client receives `hello` first, then gets back whatever it sends.
- Surprise or difficulty: the in-process test client never touches Uvicorn, so it would pass even
  if the real server could not accept WebSockets (plain Uvicorn needs a WebSocket library). We
  added `websockets` up front and confirmed with a real running server, not just the tests.
  Starlette also now deprecates `httpx` for its test client in favour of `httpx2`.
- Open question: when the web app on port 5173 starts calling REST on port 8000, should we allow
  that with CORS or hide the difference behind a Vite proxy?

### 2026-10-12 — Task 0.4 Web skeleton with three panels
- Concept: the UI is split into views that share one selection store (what the user picked); the
  TypeScript types are generated from the Python message models, so the two sides cannot drift.
- Key equation(s) in words: none, plumbing task.
- What we observed: three resizable panels and a badge that turns green only after the server's
  `hello` message arrives; stopping the Python server turns it red, restarting turns it green
  again within seconds (reconnect every 2 seconds).
- Surprise or difficulty: the first generated types were wrong in two ways (a field named
  `version` produced two clashing type names, and fields with defaults came out optional). Both
  were fixed on the Python side (a shared base model, and dropping per-field titles). Several
  tools had new major versions (TypeScript 7, Vite 8, react-resizable-panels 4), so I read their
  installed APIs instead of relying on memory. React's development mode opens each socket twice
  on purpose (it mounts, unmounts and remounts to expose cleanup bugs), which is why the server
  log shows two connections per page load.
- Open question: when the generated types are committed, what is the cheapest check that stops
  them going stale when someone edits `schemas.py`? (Task 0.6's CI is a natural place.)
