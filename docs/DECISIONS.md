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
- **Status:** accepted at task 2.5 (2026-10-19) · **Decision:** ARCHITECTURE §6.4.
  **Why:** same inputs for every method; matches the reference video; all normalised.
  **Frozen:** `OBSERVATION_VERSION = "v1"` and `OBSERVATION_NAMES` in `env/sensors.py`;
  `tests/env/test_contract.py` fails if the count, order, ranges or signs of the observation, or
  the clipping and sign conventions of the action (§6.5), change. Any such change needs a new ADR
  and a new version.

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

### ADR-019 — api v1.3: the neuron driver, `set_param`, the neuron readout and activation curves
- **Status:** accepted (2026-10-16)
- **Context:** ADR-018 said `set_param` would arrive with task 1.9, which builds the one-neuron driver and its view. The browser must show the neuron's inputs, weighted sum and output, and let the user change the weights live, without ever computing network outputs itself (ADR-012).
- **Decision:** (1) A new `neuron` policy choice. All cars of a scenario share one neuron: steer = f(w_left * d_left + w_right * d_right + bias), with d_left and d_right the far-left (+60 degrees) and far-right (-60 degrees) ray readings (observation entries 0 and 4), and a fixed throttle. It starts with all weights and the bias at 0 and a throttle of 0.15. (2) One command `set_param {name, value}`, where `name` is `w_left`, `w_right`, `bias`, `throttle` or `activation`; the value is a number, or one of tanh, sigmoid, relu, linear for `activation`. Weights and bias are limited to +-30 and the throttle to [-1, 1]; anything else is an `error`. The server answers with a frame. (3) `frame.selected` gains `neuron`: the selected car's inputs, which rays they are, weights, bias, each input's contribution (weight times input), the weighted sum z, the output, the activation and the throttle, all computed by Python. (4) `GET /api/activations` returns z values from -5 to 5 with the four activation functions sampled on them, so the browser can draw the activation curve without computing it. The API version becomes **v1.3**.
- **Alternatives:** (a) the browser recomputes z and the activation for instant slider feedback: smoother, but it breaks ADR-012 and the display could disagree with the car; (b) separate typed commands such as `set_weight` and `set_activation`: stricter, but more messages than the one name the contract lists; (c) start with weights that already steer: removes the experiment of finding them, which is task 1.10's job.
- **Consequences:** a slider drag shows its effect after one round trip to the server (about one frame). The generated TypeScript types grow. Per-car neurons and richer networks arrive with Phase 2 and extend `selected` then.

### ADR-020 — The neuron driver's settings last for the whole live session
- **Status:** accepted (2026-10-18)
- **Context:** ADR-019 created a fresh all-zero neuron on every `load_scenario`. Reset already kept the weights, but changing the number of cars, the start, the track or the driver wiped them, and so did a server restart. The user asked for the weights and settings to stay put.
- **Decision:** each live session keeps one neuron. It starts at zeros (as before) and keeps its weights, bias, activation and throttle through `reset` and every `load_scenario`, including a detour through another driver. The browser remembers the settings it last saw in a neuron readout and, after a reconnect (a new session), sends them back with `set_param`. No message or field changes, so the API stays **v1.3**.
- **Alternatives:** (a) keep the settings only through `reset` (the old behaviour): wipes hand-tuned weights on any scenario change; (b) store the settings on the server across sessions: two browser tabs would then fight over one neuron; (c) also keep them across a page reload with browser storage: not asked for, possible later.
- **Consequences:** a reloaded page still starts from zeros. Saving and loading named settings (for example `configs/baselines/hand_neuron.yaml`) would be a later feature.

### ADR-021 — Local secret scan with detect-secrets, run through Python; gitleaks stays in CI
- **Status:** accepted (2026-10-18)
- **Context:** ADR-014 put gitleaks in the pre-commit hooks. Windows Smart App Control now blocks the compiled `gitleaks.exe` that pre-commit builds ([WinError 4551]), so every commit was refused. The `python -m` workaround used for the private-key hook cannot help, because gitleaks is not a Python program. Task 0.6 named "run that check in CI instead" as the fallback, which would leave no secret scan before a commit.
- **Decision:** the local secret scan is detect-secrets (v1.5.0, Yelp), a Python scanner, started as `python -m detect_secrets.pre_commit_hook` with a baseline file `.secrets.baseline` (no findings in the repository today). Lock files and the generated TypeScript types are excluded, as their hashes look like secrets. gitleaks keeps scanning the whole history in CI on every push, and GitHub push protection stays on.
- **Alternatives:** (a) rebuild the gitleaks hook: a large download, and Smart App Control may block the new program too; (b) gitleaks in CI only: no secret scan before a commit; (c) turn Smart App Control off: Windows cannot turn it back on without a reinstall.
- **Consequences:** two scanners with different rules guard different stages. Checked when installed: the hook flags a planted fake GitHub token and AWS key, and passes on every file in the repository. If detect-secrets ever reports a harmless string, it is reviewed and recorded in `.secrets.baseline`, never skipped.

### ADR-022 — Per-track time limit: 2 laps at an average of 16 m/s
- **Status:** accepted (2026-10-18)
- **Context:** the time limit was a fixed 90 s (ARCHITECTURE §6.6, Driving Exam v1), chosen for the 731 m `train_loop_01`. The half-size Monza tracks (task 1.11) are 2,900 m a lap; 2 laps in 90 s would need 64 m/s, over twice the top speed. A time limit is needed so that episodes end, costs stay bounded and faster driving scores better, but its value only has to be fair to the track.
- **Decision:** time limit = ceil(target laps × lap length / 16 m/s), in whole seconds (`env/world.py`: `time_limit_for`, `WorldConfig.for_track`). 16 m/s is a little over half the 30 m/s top speed. It gives 92 s on `train_loop_01` and 363 s on Monza. The Driving Exam v1 and the live server use it; a bare `WorldConfig()` keeps 90 s, so existing tests and the recorded Study S1 (run at 90 s) are unchanged.
- **Alternatives:** (a) one fixed limit long enough for every track (for example 400 s): a car could crawl round the short loop and still pass; (b) a limit per track written into each track file: a track contract change, and every new track needs a hand-chosen number; (c) a time limit set by a reference driver's lap time: depends on a driver that does not exist yet.
- **Consequences:** `train_loop_01` gets 2 more seconds than before (92 s versus 90 s). Lap times stay comparable within a track, not across tracks. Fitness v1's time bonus (time limit − finish time) grows with the track, which is fine while fitness is compared on one track at a time.


### ADR-023 — api v1.4: the `random_networks` policy
- **Status:** accepted (2026-10-19)
- **Context:** task 2.6 puts multilayer networks in the live session: the roadmap's scenario "20 random networks, seed selectable". ADR-016 says adding to the message contract bumps the API minor version.
- **Decision:** (1) `load_scenario.policy` and `scenario.policy` gain the value `random_networks`: car i is driven by its own [7, 16, 16, 2] network with Xavier-initialised weights, zero biases and tanh everywhere. (2) The networks come from the scenario `seed` (seed 0 when it is null), through the child stream named "init" of the seed's generator, so a seed always gives the same drivers. The same seed also keeps its old meaning of a seeded random start; with no seed every car starts on the start line, so differences between cars come only from their networks. (3) The networks are built when the scenario loads and kept through `reset`. (4) No new message or field: the frame is unchanged (`selected.neuron` is null for this policy). Sending each car's network trace to the browser is left to task 2.7. The API version becomes **v1.4**.
- **Alternatives:** (a) a separate `network_seed` field: lets starts and brains vary independently, but adds a field and a UI control for little gain now; (b) a policy where all cars share one random network: that is `MLPPolicy`, useful for training but not for showing "random brains differ"; (c) rebuild the networks on every `reset`: the same result for a given seed, but loses the chance to keep edited networks later.
- **Consequences:** the generated TypeScript types and the driver dropdown gain one value. A seeded start and the brains are tied to one number, so they cannot be varied separately until a later task adds a field.


### ADR-024 — api v1.5: the `network` message and `selected.trace`
- **Status:** accepted (2026-10-19)
- **Context:** task 2.7 draws the selected car's multilayer network in the browser. ADR-012 says the browser never computes network outputs, so the server must send every number: the weights and the values of the forward pass. ADR-019 had already said richer networks would extend `selected` in Phase 2. The roadmap lists frame size as a risk.
- **Decision:** (1) A new server message `network {car, layer_sizes, activations, weights, biases}` carries one car's weights and biases (rounded to 4 decimals). It is sent after `scenario` when a scenario loads and again after each `select_car`, because the weights do not change while driving. (2) `frame.selected` gains `trace {activations, weighted_sums}`: the selected car's forward pass on its current observation (rounded to 4 decimals; 0.6 KB per frame, measured), so the diagram's input circles hold exactly the observation. It is null for the other drivers. (3) The server computes the trace on demand for the current observation (`trace_of` on the policy) instead of reusing the trace of the last `act` call, which belongs to the previous step. (4) The Network view shows the diagram for the `random_networks` driver and keeps the Neuron view for the others. The API version becomes **v1.5**.
- **Alternatives:** (a) repeat the weights in every frame: simpler, but the weights are 3.3 KB (measured) and a 20-car frame is 1.4 KB, so every frame would be about 3.4 times larger for numbers that never change; (b) send the weights over a REST call: needs the session's networks to be reachable by id, which live sessions do not have yet; (c) send the whole population's traces in every frame: 20 times the data for something only the selected car's view needs.
- **Consequences:** a client must keep the latest `network` message and match its `car` to `frame.selected.index` (the diagram waits for the matching pair). The generated TypeScript types grow by two models. When learning changes weights during a drive (later phases), a new message or a periodic resend will be needed; this ADR covers fixed networks only.


### ADR-025 — api v1.6: live explanations in the frame and a `what_if` command (instead of REST for now)
- **Status:** accepted (2026-10-19, chosen by the user from three options)
- **Context:** task 2.10 needs, for the selected car, the contributions and sensitivity of its outputs (task 2.9) and the network's answer for a changed observation (what-if). The roadmap and ARCHITECTURE §7 named a REST route `POST explain/forward {network ref, observation}`, but a live car's network has no reference: it lives inside one WebSocket session (ADR-018, ADR-024).
- **Decision:** (1) Every frame of the random-networks driver carries `selected.explain`: both outputs' receipts and the 2 × 7 sensitivity table for the selected car's current observation (measured: 0.8 KB of a 2.8 KB 20-car frame; the whole frame takes about 0.4 ms to build). (2) A new WebSocket command `what_if {observation}` (paused only) is answered with an `explanation` message computed from the car's exact, unrounded weights, including the real outputs now and the sensitivity's straight-line prediction, so the browser does no maths (ADR-012). (3) `POST explain/forward` is deferred to Phase 5, when checkpoints give networks a real reference (run + unit). The API version becomes **v1.6**.
- **Alternatives:** (B) a stateless REST route with the weights sent back by the browser: matches the roadmap's route, but the weights it has are rounded to 4 decimals, so a what-if at the current observation would not equal what the car does (about 1e-3 off), and 3.3 KB travels with every slider move; (C) a REST route that reaches live sessions by id: exact, but needs shared global session state and locking, which ADR-018 avoided.
- **Consequences:** the roadmap's file `server/app.py` is not changed in task 2.10. What-if works only while paused (a moving car's observation changes 30 times a second, so a question about it would be stale at once). Phase 5 adds the REST route for checkpoints next to this command.


### ADR-026 — api v1.7: the what-if ghost car, and `what_if` sends only the changed inputs
- **Status:** accepted (2026-10-19, follow-up to task 2.10 at the user's request)
- **Context:** ARCHITECTURE §8.2 says that dragging a what-if slider shows a ghost car with the alternative action. The browser may not simulate (ADR-012), so the motion must come from Python. Also, `what_if` (ADR-025) sent the whole observation as the browser had it, rounded to 4 decimals, so even "unchanged" inputs differed slightly from the car's real ones.
- **Decision:** (1) Every `explanation` carries `ghost`: the server runs `env/car.step_cars` for 2 s (60 steps) from the selected car's pose, as two cars in one batch, one holding the network's real action and one the what-if action (both clipped to [-1, 1]); it sends both paths and their end poses. Only the car physics runs: no crash, track or progress rules. (2) The look-ahead is 2 s: at the slow speeds of random brains (about 2 m/s) 1 s left the ghost under the real car. (3) `what_if` now sends `overrides {index: value}` instead of the full observation; the server applies them to the car's exact observation, so `{}` gives exactly the car's real output. The real and what-if actions are both computed by the same `forward` call, so equal inputs give equal bits. (4) The Network page's car switcher gains a drop-down of every car. The API version becomes **v1.7**.
- **Alternatives:** (a) let the ghost re-run the network at every step (closed loop): truer to driving, but the what-if question is about one changed observation, which no longer holds after the first step; (b) a longer look-ahead such as 3 s: clearer at low speed but a held action drifts further from what the network would really do; (c) keep sending the full observation and snap values within rounding of the real ones: works, but hides the rounding rather than removing it.
- **Consequences:** a ghost path can leave the road (the world's crash rule is not applied), which the legend explains as "held actions only". The what-if command is not backward compatible with v1.6 (`observation` is rejected); the browser and the generated types change together.


### ADR-027 — api v1.8: the decision trail and looking back at an earlier step
- **Status:** accepted (2026-10-20, task 2.11; options chosen by the user: A1 + B1)
- **Context:** ARCHITECTURE §8.2 asks for a trail behind the selected car, coloured by the selected output, and for clicking a trail point to "jump to that step" and show the network trace then. The browser may not compute traces (ADR-012), and a live session so far kept no history.
- **Decision:** (1) **Look, do not rewind (A1):** clicking a trail point or dragging "Look back" (paused only) sends `inspect_step {step}`; the server answers with `moment`: the car's pose, rays, observation and action at that step, and its network's trace and explanation recomputed from the stored observation by the same functions a frame uses. The simulation stays at its current step; Step, Play, another car or a new scenario returns the views to now. (2) **Memory (B1):** the session remembers the last 60 s (1,800 steps) of every car's pose, speed, observation (full precision), action and running flag in fixed-size arrays (step t in row t mod 1,800), about 3.8 MB for 20 cars and 19 MB for 100. (3) **Trail on the wire:** each frame's `selected.trail` carries the selected car's new points (one per step taken since the last frame); after a load, a reset or `select_car` it carries every remembered point with `replace: true`. Only steps at which the car was still running are listed. `scenario.trail_seconds` tells the browser how much to keep. (4) The API version becomes **v1.8**.
- **Alternatives:** (A2) rewind the world by re-simulating from step 0 to t (it is deterministic): a truer jump, but slow late in a long drive, it throws away the future and the what-if ghost and frames would need rework; (B2) the whole episode for the selected car only: tiny, but a newly selected car has no trail; (B3) the whole episode for every car: up to about 100 MB for 100 cars on Monza; a separate `trail` message instead of a frame field: cleaner to read but would change the message sequence every existing client and test relies on.
- **Consequences:** a trail point older than 60 s cannot be inspected (the slider starts at the oldest remembered step). The HUD kept showing the present while an earlier step was looked at (changed by ADR-028); the Network view, the drawer, the inspector and the rays show the earlier step, with a banner and "Back to now". What-if is unavailable while looking back (it asks about the present). The neuron driver's readout is not part of a moment, because its settings may have changed since.


### ADR-028 — api v1.9: the HUD shows the step being looked at
- **Status:** accepted (2026-10-20, follow-up to task 2.11 at the user's request)
- **Context:** with ADR-027 the HUD kept showing the present (speed, status, action, progress, step, rays) while the rays, the outlined car and the Network view showed an earlier step, so the screen mixed two moments.
- **Decision:** the session also remembers every car's progress and status code per step, and `moment` gains `progress` and `status`. While looking back, every HUD number comes from the moment: speed, status, steer and throttle, step and time, progress and laps, and the five rays; a highlighted line says which step is "now". The steer and throttle shown are the action the car chose at that step (what the Network view shows there); live, the HUD keeps showing the last action applied, as before. The API version becomes **v1.9**.
- **Alternatives:** (a) hide the HUD while looking back: no mix-up, but the numbers a learner wants next to the network would be gone; (b) recompute progress in the browser from the trail: not allowed (ADR-012) and not exact.
- **Consequences:** about 8 % more memory for the drive memory (4.1 MB for 20 cars, 20 MB for 100; progress as a float, status as one byte per car and step). A status banner (crashed, finished) disappears while a step before the end is looked at, because the car was still running then.
