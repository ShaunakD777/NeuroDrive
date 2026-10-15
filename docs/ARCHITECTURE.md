# NeuroDrive — Architecture

Status: v1 (planning). Changes to §6 contracts require an ADR in `DECISIONS.md` and a version bump.

## 1. Principles

1. **Python core is the single source of truth** for physics, sensors, network maths, learning and
   explanation maths. The browser only displays and sends commands. (ADR-001, ADR-012)
2. **Learnability over cleverness.** Core algorithms are written from scratch with NumPy array
   arithmetic only. No ML frameworks in `src/`. (ADR-006)
3. **Reproducibility by construction.** Fixed time step, explicitly seeded generators, run folders
   that capture config + seed + code version. A checkpoint + track + seed reproduces a drive exactly.
4. **Separation of responsibilities.** The environment does not know about networks. Learners do
   not know about the UI. Views do not know how a network was trained.
5. **Batched by default.** The environment and the population forward pass operate on N cars at
   once (NumPy vectorisation). (ADR-008)

## 2. Technology stack

| Concern | Choice | From scratch? |
|---|---|---|
| Language (core) | Python 3.12, managed with `uv` | — |
| Array arithmetic | NumPy | Library (arithmetic only) |
| Neural network (forward, backprop, init, save/load) | Our code | **Yes** |
| Losses and optimizers (SGD, Momentum, RMSProp, Adam) | Our code | **Yes** |
| Random search, GA, ES, NEAT | Our code | **Yes** |
| REINFORCE, actor-critic, PPO | Our code | **Yes** |
| Explanations (contributions, sensitivity, response curves) | Our code | **Yes** |
| Car physics, track geometry, raycasts | Our code (simple 2D kinematic model) | Yes (plumbing, written by Claude) |
| Config files | YAML + Pydantic validation | Library |
| CLI | Typer (`nd` command) | Library |
| Backend server | FastAPI + Uvicorn (REST + WebSocket) | Library |
| Run storage (master copy) | Plain files: YAML, JSON, JSON-lines, `.npz` | — |
| Run index for querying | SQLite via Python's built-in `sqlite3` (mirror of run folders, ADR-007) | Our own SQL |
| Version control | Git + GitHub; phase branches and tags (ADR-014) | — |
| Secret and hygiene checks | pre-commit hooks: gitleaks, private-key detection, large-file check, ruff | Tools |
| Continuous integration | GitHub Actions: lint, tests, secret scan on every push | Tool |
| Python tests | pytest (+ Hypothesis for geometry property tests) | Library |
| Lint/format | ruff | Library |
| Web app | Vite + React + TypeScript | Library (display only) |
| Web state | Zustand | Library |
| Track drawing | HTML Canvas 2D | Our drawing code |
| Network diagram | SVG (every neuron/edge clickable) | Our drawing code |
| Charts | uPlot | Library |
| Web→Python type sharing | Pydantic JSON Schema → `json-schema-to-typescript` | Generated |
| Web tests | Vitest (logic), Playwright (smoke) | Library |
| Optional cross-checks | PyTorch in `reference/` only | Library, never imported by `src/` |

## 3. Processes at runtime

```
┌───────────── Browser (web/) ─────────────┐
│ Environment view │ Network view │ Experiment view
│        └── Selection store + playback ──┘ │
└──────────────▲──────────────┬────────────┘
     frames,    │ REST + WS     │ commands
     recordings │               ▼
┌──────────── Python server process (nd serve) ────────────┐
│ live sessions (World + policies + clock) │ REST handlers  │
│ run watcher (tails runs/*/metrics.jsonl) │ explain calls  │
└──────────────▲───────────────────────────────────────────┘
               │ spawns / reads files
┌──────────── Training worker processes (nd train) ────────┐
│ Learner + Runner → writes runs/<run_id>/...               │
└──────────────────────────────────────────────────────────┘
```

- **Live mode:** the server steps a World in real time (× speed factor) and pushes frames over a
  WebSocket.
- **Replay mode:** the browser requests a full Episode Recording once (re-simulated
  deterministically from checkpoint + track + seed), then scrubs it locally. No per-frame round
  trips.
- **Training** runs in separate processes, launched from the CLI or the UI. They only write files.
  The server watches the files and pushes progress to the browser.

## 4. Repository layout

```
neurodrive/
├─ CLAUDE.md                     instructions for Claude Code
├─ README.md
├─ .gitignore / .gitattributes   what never enters git; consistent line endings
├─ .env.example                  placeholder names for any future secret (real .env is ignored)
├─ .pre-commit-config.yaml       secret scan, key detection, large files, ruff (task 0.6)
├─ .github/workflows/ci.yml      lint + tests + secret scan on every push (task 0.6)
├─ .claude/                      project slash commands + settings (committed); settings.local.json ignored
├─ pyproject.toml                Python package + deps (uv)
├─ configs/                      YAML experiment configs (one per method/study)
├─ tracks/                       track definition JSON files
├─ runs/                         run folders (git-ignored)
├─ src/neurodrive/
│  ├─ core/                      maths with no knowledge of driving
│  │  ├─ rng.py                  seeded generator creation and child-seed spawning
│  │  ├─ activations.py          tanh, sigmoid, relu, linear (+ derivatives from Phase 3)
│  │  ├─ neuron.py               single neuron (Phase 1 teaching object)
│  │  ├─ nn/
│  │  │  ├─ spec.py              NetworkSpec: layer sizes, activations, versioning
│  │  │  ├─ network.py           parameters, clone, flatten/unflatten (genome), parameter count
│  │  │  ├─ init.py              initialisation schemes
│  │  │  ├─ forward.py           forward pass + ForwardTrace, single & batched
│  │  │  ├─ population.py        forward pass for N different networks at once
│  │  │  ├─ backward.py          backpropagation (Phase 3)
│  │  │  ├─ gradcheck.py         finite-difference gradient checking
│  │  │  └─ io.py                save/load .npz network files
│  │  ├─ losses.py               MSE, Huber (+ derivatives)
│  │  └─ optim/                  sgd.py, momentum.py, rmsprop.py, adam.py, base.py
│  ├─ env/                       the driving world (knows nothing about networks)
│  │  ├─ track.py                load/derive track geometry, projection, progress
│  │  ├─ car.py                  batched kinematic car physics
│  │  ├─ sensors.py              batched raycasts + observation builder
│  │  ├─ world.py                World: reset/step for N cars, termination
│  │  ├─ feedback.py             the three taps: teacher label, step reward, episode summary
│  │  ├─ experts.py              scripted expert driver (privileged state)
│  │  ├─ policies.py             Policy protocol + Keyboard/Neuron/MLP/Expert/Stochastic policies
│  │  └─ episode.py              episode runner + EpisodeRecording
│  ├─ learn/
│  │  ├─ base.py                 Learner contract, budget counters
│  │  ├─ supervised/             datasets.py, demos.py, imitation.py, dagger.py, toy_tasks.py
│  │  ├─ evolution/              random_search.py, ga.py, es.py, neat/ (genome, species, ...)
│  │  ├─ rl/                     returns.py, gaussian_policy.py, reinforce.py, actor_critic.py, ppo.py
│  │  └─ hybrid/                 distill.py, pretrain_finetune.py
│  ├─ experiment/
│  │  ├─ config.py               Pydantic models for every config file
│  │  ├─ runner.py               training loop orchestration, checkpoint & exam schedules
│  │  ├─ store.py                run folder read/write
│  │  ├─ db.py                   SQLite mirror: schema, insert/update, rebuild, check
│  │  ├─ exam.py                 the Driving Exam
│  │  ├─ metrics.py              driving metric definitions
│  │  ├─ stats.py                median, IQR, bootstrap confidence intervals
│  │  └─ compare.py              aligned comparisons between runs
│  ├─ explain/
│  │  ├─ contributions.py        weight × activation breakdowns, top paths
│  │  ├─ sensitivity.py          d(output)/d(input): finite differences, later backprop
│  │  └─ response.py             response curves across checkpoints
│  ├─ server/
│  │  ├─ app.py                  FastAPI app, routes
│  │  ├─ schemas.py              Pydantic message models (source for TS types)
│  │  ├─ live.py                 live sessions over WebSocket
│  │  └─ runs_watch.py           progress streaming for active runs
│  └─ cli.py                     `nd serve | train | sweep | exam | demo-record | ...`
├─ tests/                        mirrors src/neurodrive (core/, env/, learn/, experiment/, explain/, server/)
├─ reference/                    optional PyTorch cross-checks (never imported by src)
├─ scripts/                      one-off scripts: schema export for type generation, track generation, study figure export
├─ web/
│  ├─ e2e/                       Playwright smoke tests (playwright.config.ts starts both servers)
│  ├─ scripts/                   gen-types.mjs: Pydantic JSON Schema → TypeScript (`npm run gen:types`)
│  ├─ src/api/                   REST client, WebSocket client, generated types (committed)
│  ├─ src/nav/                   the views table and hash routing (#/environment, #/network, #/experiment)
│  ├─ src/state/                 selection store, layout store (Page/Split/All, drawers), playback clock
│  ├─ src/views/environment/     canvas renderer, trails, HUD, population mode
│  ├─ src/views/network/         SVG diagram, output inspector, what-if panel, diff
│  ├─ src/views/experiment/      run list, timeline, charts, comparison page, toy lab
│  └─ src/components/            shared UI pieces
└─ docs/                         concept, architecture, roadmap, checklist, protocol, journal ...
```

## 5. Layering rules

- `core/` imports only NumPy and the standard library.
- `env/` imports `core/rng.py` only (not `core/nn`). Policies are passed in as objects that follow
  the Policy protocol, so the world never needs to know what a network is. `env/policies.py` may
  import `core/nn` because adapters live there.
- `learn/` imports `core` and `env`.
- `experiment/` imports `core`, `env`, `learn`.
- `explain/` imports `core` and `env`.
- `server/` imports everything. No other module imports `server/` or FastAPI.
- `web/` talks to Python only over HTTP/WebSocket.

## 6. Contracts (versioned)

### 6.1 Units and coordinates
World coordinates in metres, x to the right, y up. Heading ψ in radians, 0 = +x, counter-clockwise
positive. The renderer flips y for the screen. Time in seconds. Fixed physics step **dt = 1/30 s**.

### 6.2 Track format (track v1)
A JSON file in `tracks/` with:
- `name`, `version`, `split` (`"train"` or `"heldout"`)
- `centerline`: list of [x, y] points forming a closed loop (last point not repeated)
- `width` (metres, default 12)
- `start`: arc-length position `s0` and direction (`+1` along the centerline order)

Derived at load time (in `env/track.py`):
- centreline resampled at 1 m spacing and smoothed (Catmull-Rom)
- cumulative arc length, total length `L_track`, unit tangent and tangent angle at each sample
- left and right boundaries offset by ± width/2, stored as line segments for raycasting

### 6.3 Car physics (physics v1)
Per car state: x, y, ψ, speed v ≥ 0, alive flag, cumulative progress P, laps.
Parameters (initial values, tuned in task 1.3 and recorded in config): wheelbase `Lw = 2.5 m`,
max speed `v_max = 30 m/s`, acceleration `a_acc = 8 m/s²`, braking `a_brk = 16 m/s²`, drag
`k = 0.1 /s`, max steering angle `δ_max = 0.5 rad`, car half-width `1.0 m`, maximum sideways
acceleration `a_lat_max = 9 m/s²` (the grip limit, ADR-003).

Each step, in this order (semi-implicit Euler):
1. Longitudinal acceleration: if throttle u ≥ 0 then a = u·a_acc, else a = u·a_brk.
2. Speed: v ← clamp(v + (a − k·v)·dt, 0, v_max). No reversing.
3. Heading with grip limit: requested curvature κ_req = tan(steer·δ_max) / Lw; allowed curvature
   κ_max = a_lat_max / max(v², 1); actual κ = κ_req clamped to [−κ_max, +κ_max]; then
   ψ ← ψ − v·κ·dt. **Positive steer turns right.** (At 30 m/s the tightest possible turn has a
   radius of 100 m; at 10 m/s, about 11 m. Asking for more makes the car run wide.)
4. Position: x ← x + v·cos ψ·dt, y ← y + v·sin ψ·dt.
5. Only cars that are alive update. Dead cars freeze.

### 6.4 Sensors and Observation (observation v1, 7 numbers)
- 5 rays at angles relative to heading: +60°, +30°, 0°, −30°, −60° (order: far-left → far-right).
- Ray reading = distance to the first boundary hit, capped at `R = 50 m`, divided by R → [0, 1].
- `o1..o5` = the 5 ray readings (d1 far-left … d5 far-right)
- `o6` = v / v_max  → [0, 1]
- `o7` = heading error / π → (−1, 1], where heading error = wrap(ψ − tangent angle at the car's
  projection onto the centreline). Positive means the car points left of the track direction.

Note: `o7` uses track knowledge (like the reference video's θ). This is documented and identical
for every method.

### 6.5 Action (action v1, 2 numbers)
- `steer` ∈ [−1, 1]: −1 = full left, +1 = full right
- `throttle` ∈ [−1, 1]: +1 = full acceleration, −1 = full braking
- The environment clips anything outside the range. Networks produce actions with a tanh output layer.

### 6.6 Projection, progress, termination
- **Projection:** the nearest point on the centreline, found by searching a window around the
  previous index (global search at reset). It gives arc length `s` and lateral offset `e`
  (positive = left of centreline).
- **Progress:** Δs wrapped into (−L_track/2, L_track/2]; cumulative progress P += Δs (driving
  backwards reduces P); laps = floor(P / L_track).
- **Crash:** |e| > width/2 − car half-width.
- **Episode ends** for a car when it crashes, reaches the time limit (config; default 90 s), stalls
  (progress < 2 m over the last 3 s, checked after the first 2 s), or completes the target laps
  (config; default 2).

### 6.7 Feedback taps (feedback v1)
- **Teacher label** `teacher(state) → action`. The scripted expert sees the *privileged full
  state* (position on the centreline, curvature ahead). The student sees only the 7-number
  observation. That gap is itself a lesson.
  - Steering: pure pursuit toward a look-ahead point on the centreline at distance
    ℓ = 8 m + 0.5·v ahead.
  - Throttle: target speed = min(v_max, sqrt(a_lat_max / κ_ahead)), where κ is the curvature and
    a_lat_max = 9 m/s². Accelerate or brake proportionally toward the target.
- **Step reward** (reward v1): r_t = 0.1·Δs_t (metres of forward progress this step); add −5 on
  the crash step. Revised in Phase 10 via ADR.
- **Episode summary** (summary v1): total progress P, laps, crashed (yes/no), steps alive, lap
  times, mean speed, steering smoothness. **Fitness v1** = P + 200·(laps completed), plus a time
  bonus of (time_limit − finish_time) if the target laps were completed. Examined in Phase 7.

### 6.8 Network spec and file (network v1)
- `NetworkSpec`: `layer_sizes` (default [7, 16, 16, 2]), `hidden_activation` (default tanh),
  `output_activation` (tanh), `version`.
- Parameters: for each layer l, a weight matrix W_l of shape (n_out, n_in) and a bias vector b_l
  of length n_out.
- **Flat genome order:** W_0 row by row, then b_0, then W_1, b_1, and so on. For [7,16,16,2]:
  128 + 272 + 34 = **434 numbers**.
- File: `.npz` with arrays `W0, b0, W1, b1, ...` plus a `spec` JSON string.

### 6.9 Forward pass and ForwardTrace
z_l = W_l · a_{l−1} + b_l, then a_l = f_l(z_l), with a_0 = the observation. The **ForwardTrace**
records every a_l and z_l. It feeds the Network view, the explanations and backprop.

### 6.10 Policy protocol
A policy maps observations of shape (N, 7) to actions of shape (N, 2). It may also provide a
ForwardTrace for a chosen car index. Kinds: Keyboard, Neuron (Phase 1), MLP (one network shared
by all cars), Population (a different network per car), Expert, Stochastic (RL: Gaussian noise
around the network output, with a deterministic mode for exams).

### 6.11 Episode Recording (recording v1)
Metadata: track, seed, dt, policy reference (run, unit, car indices), physics/observation versions.
Per step, per car: x, y, ψ, v, alive, P, rays(5), observation(7), action(2).
For the **focus car(s)**: the full ForwardTrace at every step.
Population replays may store pose-only (x, y, ψ, alive) to keep size down.
Stored as `.npz`; sent to the browser as JSON with floats rounded to 4 significant digits.

*Layout as built (task 1.6, `env/episode.py`):* for T steps and N cars, state arrays (`x`, `y`,
`psi`, `v`, `alive`, `progress`, `rays`, `observation`) have T + 1 frames (frame t is the state
before action t; frame T is the final state) and `action` has T entries. The start is stored as
`start_s0`, `start_lateral`, `start_heading_offset` and `start_speed` (what `World.reset` needs to
recreate it exactly, random draws included), plus the final `status` of each car. Metadata is JSON
text inside the file and also holds the world rules (`world_config`) and `car_params`, so a replay
rebuilds the same world. Replaying the actions reproduces every array bit for bit.

### 6.12 Learner contract (learner v1)
Every learning method provides:
- `name` and `feedback_kind` ∈ {`labels`, `episode_score`, `step_reward`, `none`}
- set-up from (config, seeded generator)
- `train_unit()` — advance one natural unit (epoch / generation / policy update). Returns its
  native metrics and **budget increments**: env steps, demonstration frames, teacher queries,
  wall-clock seconds
- `current_networks()` — the network(s) to checkpoint (the best one first)
- save/restore of its internal state, for resuming

The **Runner** (not the learner) owns the loop, checkpoint schedule, exam schedule, logging and the
stop conditions (max env steps, max wall-clock, max units).

### 6.13 Run folder (run v1)
```
runs/<YYYYMMDD-HHMMSS>-<method>-<tag>/
  config.yaml        fully resolved config, including seed
  meta.json          git commit + dirty flag, Python/NumPy versions, start/end, status
  metrics.jsonl      one JSON object per unit: unit, env_steps_total, demo_frames_total,
                     wall_clock_s, native metrics (loss / fitness stats / return ...)
  checkpoints/       unit_<n>.npz (+ index.json listing units and what each holds)
  exams/             unit_<n>.json (Driving Exam results)
  extra/             method-specific: population stats, lineage, dataset references
  notes.md           optional human notes
```
Mirror: `runs/neurodrive.db` (ADR-007). Folders are the master copy; the database can always be
rebuilt from them with `nd db rebuild`. All of `runs/` is git-ignored.

### 6.14 Exam result (exam v1)
Exam spec version, list of tracks, per-episode metrics (track, start index, progress, laps,
crashed, lap times, mean speed, smoothness, time to crash), and aggregates per track and overall
(median, IQR, completion rate). See `EXPERIMENT_PROTOCOL.md`.

## 7. Server API (api v1.1, ADR-016)

REST (`/api/...`):
- `GET health` → `{status: "ok", version}` (`version` is the NeuroDrive package version)
- `GET tracks`, `GET tracks/{name}` (geometry for drawing)
- `GET runs`, `GET runs/{id}` (config, meta, metrics), `GET runs/{id}/checkpoints`
- `GET runs/{id}/checkpoints/{unit}`: network as JSON
- `GET runs/{id}/checkpoints/{unit}/diff?from={unit}`: per-weight change
- `POST runs` (launch from a config), `POST runs/{id}/stop`
- `POST replay` {run, unit, track, seed, focus_cars} → Episode Recording
- `POST explain/forward` {network ref, observation} → ForwardTrace + contributions + sensitivity (what-if)
- `POST explain/response` {run, units, output, input, range} → response curves
- `POST compare` {run_ids} → aligned exam and budget tables

WebSocket `/ws/live`:
- client → server: `load_scenario`, `play`, `pause`, `step`, `set_speed`, `reset`, `select_car`,
  `set_param` (e.g. neuron weights in Phase 1), `key_state` (manual driving)
- server → client: `hello`, `scenario` (track + cars), `frame` (all cars: pose, alive, action;
  selected car: rays, observation, ForwardTrace), `episode_end`, `error`
- `hello` is sent first, on connect: `{type: "hello", version, api}` (`api` is the message-contract
  version, currently `"v1.1"`). Until task 1.7 the socket then simply echoes any text it receives.
- Frames are throttled to at most 30 per second.
- In development the browser reaches both REST and WebSocket through the Vite dev server, which
  forwards `/api` and `/ws` to the Python server on port 8000 (ADR-015).

WebSocket `/ws/runs`: `metrics`, `checkpoint`, `exam_done`, `run_status` events.

All message shapes are Pydantic models in `server/schemas.py`. TypeScript types are generated from
them (`npm run gen:types`), so the two sides cannot drift apart.

## 8. Web app

### 8.0 Navigation and layout (ADR-017, built in task 0.4b)
- **Page layout (default):** one full page per view (Environment, Network, Experiment), each with its
  own address (`#/environment`, `#/network`, `#/experiment`), switched with tabs in the top bar.
- **Companion drawer:** every page has a collapsible side drawer showing a compact version of a
  linked view (for example a mini network beside the track), so cause and effect stay visible.
- **Layout switch:** Page (default), Split (any two views side by side, resizable) and All (the
  three panels). The choice is remembered per browser.
- The selection store (§8.1) and the live connection belong to the app, not to a page, so they
  survive page changes; the linking rules in §8.2 apply in every layout.

### 8.1 Selection store (single source of UI truth)
`mode` (live | replay), `runId`, `unit` (checkpoint), `carId`, `neuron` {layer, index},
`connection` {layer, from, to}, `step` (time index), `output` (steer | throttle), `whatIf` (input overrides).

### 8.2 Linking behaviour

| User action | Environment view | Network view | Experiment view |
|---|---|---|---|
| Click a car | Highlight it, show its rays | Show its network with live activations | Mark its run/checkpoint |
| Click an output neuron | Colour rays by sensitivity; colour trail by that output | Contribution bars; highlight strongest paths to inputs | Show the output's response curve across checkpoints |
| Pause and drag a what-if slider | Ghost car shows the alternative action | Recompute activations (server `explain/forward`) | — |
| Click a point on the trail | Jump to that step | Show the trace at that step | — |
| Select a run and scrub the timeline | Replay that checkpoint's exam drive | Weights at that checkpoint, changes highlighted | Cursor on all charts |

### 8.3 Output-neuron inspector (built up over phases)
1. Readout: the value plus its physical meaning ("−0.93 = hard left").
2. Contributions: c_i = W[j, i] · a_i for each incoming connection, sorted; total = Σ c_i + b_j.
3. Sensitivity: ∂output / ∂input_k for each sensor (finite differences in Phase 2, backprop from Phase 3).
4. History: response curve of the output against one input at several checkpoints.

## 9. Performance plan
- Targets: live view at 30 fps with 50 cars; one GA generation (50 cars × 60 s simulated) in ≤ 5 s
  of wall-clock time.
- Steps, in order, only when profiling shows a need: vectorise everything (default); windowed
  projection search; spatial grid for raycasts; process pool for independent seeds and runs;
  Numba for env hot loops only, via an ADR.

## 10. Testing layers
See `EXPERIMENT_PROTOCOL.md` §1 for the component verification table.
- Unit tests for every maths function, against hand-computed values.
- Gradient checks for every differentiable component.
- Determinism tests for the environment, learners and runner.
- Contract tests: generated TS types match the schemas; run folders validate against the config models.
- Web: Vitest for selection/linking logic; one Playwright smoke test (three panels render,
  WebSocket connects, clicking a car switches the network view).

## 11. Repository hygiene and version control (ADR-014)

**What is committed:** source code, tests, docs, configs, track files, `.claude/commands/`,
`.claude/settings.json`, study reports and small figures (< 1 MB each).

**What is never committed:** `.env` and any `.env.*` except the placeholder files `.env.example`
and `.env.local.example`; keys and certificates
(`*.pem`, `*.key`, `*.p12`, `id_rsa*`); tokens or passwords in any form; `runs/` (run folders,
checkpoints, recordings, SQLite database); `data/` (recorded demonstrations); `node_modules/`,
build output and caches; `.claude/settings.local.json`; editor and OS files.

**Configuration and secrets:** NeuroDrive needs no secrets. If one is ever needed (for example a
token for an optional external service), it is read only from an environment variable prefixed
`ND_`, documented with a placeholder in `.env.example`, and never written into configs, code, docs,
logs, run folders or test fixtures. Config files hold only experiment settings.

**Personal data:** no absolute paths containing user names, no e-mail addresses and no machine
names in committed files. `meta.json` records the git commit and library versions, not the user
or host.

**Safety net (layers):** `.gitignore` → pre-commit hooks (gitleaks, private-key detection,
large-file check) → CI secret scan on every push → GitHub secret scanning and push protection
(enabled in the repository settings where available).

**Branches, commits and versions:**
- `main` always holds the last completed phase, with all tests and CI passing.
- Each phase is developed on `phase-<N>-<short-name>` (e.g. `phase-1-one-neuron`).
- One commit per verified task: `task X.Y: <summary>`. Docs-only changes: `docs: <summary>`.
- Completing a phase: all phase tasks ✅, full test suite green → merge into `main` (no
  fast-forward, so the phase stays visible in history) → annotated tag `phase-<N>` with a one-line
  summary → push `main`, the branch and the tag after the user confirms.
- Never force-push, never rewrite pushed history, never commit directly to `main` during a phase.
