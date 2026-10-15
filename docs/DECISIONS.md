# NeuroDrive — Architecture Decision Records

Format: **ID — Title** · Status (proposed / accepted / superseded) · Context · Decision ·
Alternatives considered · Consequences. Add new records at the bottom. Never delete one; supersede it.

---

### ADR-001 — Python core with a browser-based viewer
- **Status:** accepted (2026-10-10)
- **Context:** The user is most comfortable in Python and wants to understand the core. The three
  linked views need rich interactivity (clickable neurons, scrubbing, live charts).
- **Decision:** All simulation, maths and learning in Python (`src/neurodrive`). A FastAPI server
  exposes REST and WebSocket APIs. A Vite + React + TypeScript app renders the three views.
- **Alternatives:** (a) everything in TypeScript in the browser: simplest linking, but the core
  would not be in Python; (b) Python desktop UI (PySide6 + pyqtgraph, or Dear PyGui): one
  language, but weaker interactive visuals, harder to verify automatically, harder to share;
  (c) pygame + matplotlib: weak for linked inspection.
- **Consequences:** a client/server boundary (message schemas, generated TS types). Replay data
  is sent once and scrubbed locally. Claude Code writes the web code; the user focuses on Python.

### ADR-002 — 2D top-down simulation
- **Status:** accepted · **Decision:** 2D Canvas rendering. **Alternative:** 3D (as in the reference
  video). **Why:** 3D adds no learning value and much effort; 2D shows rays, trails and populations
  more clearly.

### ADR-003 — Own kinematic physics (with a grip limit) instead of a physics engine
- **Status:** accepted (2026-10-10, revised)
- **Context:** Cars must respond believably to steering and throttle. General physics engines (Box2D, pymunk) simulate rigid bodies: collisions, contact forces, friction and momentum. In NeuroDrive, touching a wall ends the episode and cars never interact, so most of an engine would go unused. Neither engine includes a car model, so car-like steering would still have to be written on top.
- **Decision:** Our own kinematic bicycle model, fixed dt = 1/30 s (ARCHITECTURE §6.3), applied to all cars at once as NumPy arrays. It includes a grip limit: sideways acceleration v²/R is capped at a_lat_max = 9 m/s², so at high speed the car cannot turn tightly and runs wide. This makes braking before corners necessary, which gives the throttle output a real job.
- **Alternatives:** (a) Box2D / pymunk: steps bodies one by one through Python bindings (too slow for GA/RL populations), adds solver details that make exact replay harder to guarantee, and hides the maths behind a library; (b) ready-made environments such as Gymnasium CarRacing: pixel observations instead of rays, a fixed track and reward, and it conflicts with the observation contract and the linked views.
- **Consequences:** deterministic, vectorised, fast and fully explainable; grip and steering limits are config values that can change in robustness studies (task 12.2). No skidding, drifting or car-to-car contact. Revisit with a physics v2 (our own dynamic bicycle model with tyre slip) if drifting is wanted. Car-to-car interaction would be the only strong reason to reconsider an engine.

### ADR-004 — Continuous actions: any value from −1 to +1, produced by tanh
- **Status:** accepted (2026-10-10, revised)
- **Context:** Steering and throttle must be graded (a gentle left, half throttle, light braking), not just on/off.
- **Decision:** Both outputs are continuous numbers anywhere between −1 and +1. The output layer uses tanh, which smoothly maps any weighted sum z into that range: z = 0 → 0, z = 0.2 → 0.197, z = 0.5 → 0.462, z = 1 → 0.762, z = −1 → −0.762. It only gets close to ±1 when z is beyond about ±3, and never reaches it exactly. Steer: −1 full left, 0 straight, +1 full right, values between are proportional (−0.3 = 30 % of full left lock). Throttle: +1 full acceleration, 0 coast, −1 full braking, values between are proportional.
- **Alternatives:** (a) discrete actions (left/straight/right): coarse, turns control into classification; (b) sigmoid rescaled as 2·sigmoid(z) − 1: mathematically the same as tanh(z/2), so no benefit; (c) linear output with clipping: outputs can grow without limit and the gradient becomes zero once clipped.
- **Consequences:** matches the reference video and works for imitation (regression), evolution and Gaussian policy-gradient RL (the tanh output is the mean of the Gaussian, and the sampled action is clipped). Untrained networks drive roughly straight. The environment still clips actions to [−1, 1]. Watch for saturation: large weighted sums push outputs near ±1, which gives all-or-nothing "bang-bang" driving and near-zero gradients (slope 1 − 0.99² ≈ 0.02). The exam's smoothness metric measures this.

### ADR-005 — Observation contract v1 (5 rays + speed + heading error)
- **Status:** proposed → accepted at task 2.5 · **Decision:** ARCHITECTURE §6.4.
  **Why:** same inputs for every method; matches the reference video; all normalised.

### ADR-006 — From-scratch rule: NumPy yes, ML frameworks no
- **Status:** accepted · **Decision:** no torch/tf/jax/sklearn/autograd in `src/`. NumPy allowed for
  array arithmetic. PyTorch allowed only in `reference/` for cross-checks.

### ADR-007 — Run storage: plain folders (master copy) + SQLite mirror
- **Status:** accepted (2026-10-10, revised; supersedes the files-only version)
- **Context:** While learning, runs must be easy to open and read by hand. Later (large comparisons, the final report, dashboards) we need to query across many runs, e.g. "median held-out exam score of all GA runs with mutation rate 0.05".
- **Decision:** Two stores with separate roles. (1) Run folders (YAML, JSON, JSON-lines, .npz; ARCHITECTURE §6.13) are the master copy and are written first. (2) The SQLite database `runs/neurodrive.db` is a mirror built from the folders, with tables `runs`, `metrics` (one row per metric value), `checkpoints` (including a weights BLOB copy), `exams`, `exam_episodes` and `studies`. The Runner updates the mirror after each unit, with WAL mode so the server can read while training writes. `nd db rebuild` regenerates the database from the folders and `nd db check` reports differences. If they disagree, the folders win. It uses Python's built-in `sqlite3`: no new dependency, no ORM, readable SQL.
- **Alternatives:** (a) folders only: cross-run queries are slow and manual; (b) SQLite only: opaque while learning, and one bug could corrupt everything; (c) MLflow / Weights & Biases: an extra service that hides the structure.
- **Consequences:** a new module `experiment/db.py` and a new task 5.5b. Tests must show that a database rebuilt from the folders matches the one written step by step. Because `runs/` is git-ignored, the database stays out of git.

### ADR-008 — Batched environment and population forward pass
- **Status:** accepted · **Decision:** World and population networks operate on arrays of N cars.
  **Why:** Python loops over cars would be too slow for GA/RL; vectorised NumPy is idiomatic.
  Scalar reference implementations exist only in tests.

### ADR-009 — Scripted expert as the first teacher
- **Status:** accepted · **Decision:** pure-pursuit expert with privileged state; human keyboard
  demos added as an experiment. **Why:** unlimited, consistent labels; the gap between privileged
  teacher and observation-only student is itself a lesson.

### ADR-010 — Backpropagation before evolution
- **Status:** accepted · **Why:** derivative intuition is needed for ES, RL and the sensitivity
  explanations; GA can then be contrasted against something the user already understands.

### ADR-011 — Division of work
- **Status:** accepted (2026-10-10) · **Decision:** Claude Code writes all code, one task at a time,
  with a Brief before and a detailed plain-language Report after (WORKFLOW.md). The user
  reviews, asks questions and runs experiments.

### ADR-012 — The browser never simulates
- **Status:** accepted · **Decision:** the web app never computes physics or network outputs, even
  for what-if; it calls `explain/forward`. **Why:** one source of truth; no drift between two
  implementations.

### ADR-013 — Numba for environment hot loops
- **Status:** proposed (decide at task 7.0 only if profiling misses the target).

### ADR-014 — Version control on GitHub, with a secret-free repository
- **Status:** accepted (2026-10-10)
- **Context:** The user wants a versioned history on GitHub, with a saved version at the end of every stage. The repository must always be safe to publish: no secrets, API keys, tokens, private keys or personal data.
- **Decision:** One GitHub repository. Work for each phase happens on a branch `phase-<N>-<short-name>`; every verified task is one small commit; when a phase is complete it is merged into `main` and tagged `phase-<N>` (the "version" of that stage), then pushed after the user confirms. `main` always holds the last completed phase, with all tests passing. Secret safety is layered: (1) the project needs no secrets, and any future one is read only from environment variables, with a `.env` file that is git-ignored and a committed `.env.example` containing placeholders; (2) `.gitignore` excludes `.env*` (except the example), keys, certificates, `runs/`, databases, recordings, `node_modules/` and build output; (3) pre-commit hooks run gitleaks (secret scanning), private-key detection and a large-file check before every commit; (4) GitHub Actions CI repeats the secret scan plus lint and tests on every push; (5) GitHub secret scanning and push protection are switched on in the repository settings where available; (6) Claude Code is denied reading `.env` files.
- **Alternatives:** (a) commit straight to `main`: simpler, but an unfinished phase would break the "last good version"; (b) tags only, without phase branches: fine for a solo project, but harder to abandon a failed approach; (c) relying only on `.gitignore`: does not catch a key pasted into source code.
- **Consequences:** a new task 0.6 (pre-commit, CI, first push). Claude Code stages files by name after checking `git status`, never uses `git add -A` or `git add .`, never force-pushes, and never rewrites history that has been pushed. If a secret is ever committed, it must be treated as leaked: revoke it first, then clean the history.


### ADR-015 — The web dev server forwards `/api` and `/ws` to the Python server
- **Status:** accepted (2026-10-12)
- **Context:** The browser loads the page from the Vite dev server (port 5173) but the Python server listens on port 8000. A browser treats these as two different origins and blocks REST calls between them unless the Python server explicitly allows it (CORS). Task 0.3 left this question open.
- **Decision:** In development, Vite forwards every request under `/api` and `/ws` to `127.0.0.1:8000` (configured only in `web/vite.config.ts`). The browser talks to a single origin, and the web app builds its WebSocket address from the page's own address. The Python server needs no CORS configuration.
- **Alternatives:** (a) enable CORS in `server/app.py`: also works, but adds server configuration that must stay in step with the web app's address and loosens what the server accepts; (b) hard-code `localhost:8000` in the web app: couples the web code to one development setup.
- **Consequences:** `nd serve` must be running for `/api` and `/ws` to work in development (Vite logs a proxy error when it is not). If the app is ever served by the Python server itself, no web code changes are needed because the addresses are already relative. Reversible: removing the proxy and adding CORS touches only `vite.config.ts` and `server/app.py`.

### ADR-016 — api v1.1: the `health` and `hello` message shapes are part of the contract
- **Status:** accepted (2026-10-12)
- **Context:** ARCHITECTURE §7 (api v1) named `GET health` and the `hello` message but did not define their fields. Tasks 0.3 and 0.4 built them, and task 0.5 wrote the built shapes into §7. The user asked for that addition to be treated as a contract change.
- **Decision:** `GET /api/health` returns `{status: "ok", version}` and `/ws/live` sends `{type: "hello", version, api}` first on connect, where `version` is the NeuroDrive package version and `api` the message-contract version. The API version becomes **v1.1** (`API_VERSION` in `server/schemas.py`). Versioning rule for the API from now on: adding a message or a field bumps the minor number (v1.1 → v1.2); removing or changing the meaning of a field bumps the major number (v1 → v2).
- **Alternatives:** (a) keep "v1" because nothing has been released yet: simpler, but the version would stop being a reliable record of what the contract says; (b) jump to v2: overstates an additive change.
- **Consequences:** the web app can compare `hello.api` with the version it was built for. The generated TypeScript types are unchanged (`api` stays a string).

### ADR-017 — Views as pages, with a companion drawer and a layout switch
- **Status:** accepted (2026-10-12)
- **Context:** Task 0.4 put the three views (Environment, Network, Experiment) side by side as resizable panels. Even as placeholders they felt crowded, and the real content (a track canvas, a network diagram, charts) needs more room. But the project's core teaching moment is seeing a car and its network change together (§8.2 linking), which pure pages would hide.
- **Decision:** By default each view is a full page with its own address, switched with tabs. Each page has a collapsible companion drawer that shows a compact version of a linked view. A layout switch offers Page (default), Split (any two views) and All (the three panels). The selection store and the live connection live at app level, so selections survive page changes. Built as task 0.4b.
- **Alternatives:** (a) plain pages with a split option: simplest, but car and network are not visible together by default; (b) task-based workspaces ("Drive & inspect", "Network deep-dive", "Experiments"): strongest for teaching, but the most layout code and less obvious naming; (c) keep the panels with a full-screen "focus" button: smallest change, still crowded by default.
- **Consequences:** a new task 0.4b before 0.6. Later view tasks (1.8, 1.9, 2.7) also provide the compact companion version of their view. The Playwright smoke test changes from "three panels" to "pages, drawer and layouts".

### ADR-018 — api v1.2: live-session messages (column-array frames, track geometry over REST, one session per connection)
- **Status:** accepted (2026-10-15)
- **Context:** ARCHITECTURE §7 named the `/ws/live` messages but not their fields. Task 1.7 builds the live session, so the shapes must be fixed, and ADR-016 says adding messages bumps the API minor version.
- **Decision:** (1) A `frame` carries the cars as **column arrays** (`x`, `y`, `psi`, `v`, `alive`, `progress`, `status`, `action`: one entry per car), plus `selected` (rays in metres and the 7-number observation of the selected car, or null). (2) The `scenario` message does **not** carry track geometry; the browser gets it from `GET /api/tracks/{name}` (and the names from `GET /api/tracks`). (3) Each WebSocket connection owns **its own session**; it ends when the connection closes. Messages: client to server `load_scenario {track, policy, n_cars, seed}`, `play`, `pause`, `step`, `set_speed {speed}`, `reset`, `select_car {index}`, `key_state {steer, throttle}`; server to client `hello`, `scenario`, `frame`, `episode_end`, `error`. `set_param` is added with task 1.9. The API version becomes **v1.2**. The server steps the world in fixed 1/30 s physics steps; the speed factor changes how many steps are taken per second, never the step size; while playing, one frame is sent per max(1, speed) steps, i.e. at most 30 per second on average. Wire numbers are rounded (positions and speeds to 3 decimals, angles to 4).
- **Alternatives:** (a) a list of car objects per frame: easier to read, about twice the size, clumsier to loop over; to be revisited if readability matters more than size. (b) track geometry inside `scenario`: one message instead of two requests, but the geometry is resent on every scenario load. (c) one shared global session: allows spectators, but needs locking and a rule for who drives.
- **Consequences:** the browser makes a REST call for the track before drawing. The generated TypeScript types grow (every message and the track models). The track folder is fixed to the repository's `tracks/` folder (no environment variable). A live session is not recorded yet, so a live drive cannot be replayed until a later task adds that.
