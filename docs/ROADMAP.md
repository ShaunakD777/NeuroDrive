# NeuroDrive — Implementation Roadmap

Each task is small, independently verifiable, and teaches one concept. Work one task at a time
(see `WORKFLOW.md`). Status lives in `CHECKLIST.md`. File paths are relative to `src/neurodrive/`
unless they start with `web/`, `tests/`, `configs/`, `tracks/` or `docs/`.

Legend: **⚑** = a decision is needed with the user before implementing. **(opt)** = optional.

**Versions:** every phase is built on its own branch `phase-<N>-<short-name>`, one commit per
verified task. When a phase is complete, `/finish-phase <N>` merges it into `main`, tags it
`phase-<N>` and pushes after you confirm (ADR-014, ARCHITECTURE §11). The "Phase done when"
criteria below always include that release step.

---

## Phase 0 — Foundations and reproducibility

**Goal:** an empty but working project: Python package, tests, server, web app with three panels,
seeded randomness, and a safe, versioned GitHub repository.
**Learn:** why reproducibility matters; what seeds are; what a unit test proves; how version
control keeps every stage recoverable.
**Depends on:** nothing. **Enables:** everything.
**Phase done when:** `uv run pytest` passes, `nd serve` and `npm run dev` start, the browser shows
three panels and a "connected" indicator, the determinism test passes, pre-commit hooks and CI are
green, and `main` is pushed with tag `phase-0`.
**Risks:** over-building the skeleton. Keep it minimal.

**Before 0.1 (user):** create the GitHub repository (empty, no auto-generated README or
.gitignore), clone it, copy these planning files in, and make the first commit on `main`
(`docs: planning documents`). Claude Code then creates branch `phase-0-foundations`.

**0.1 Python project skeleton**
- Concept: packages, virtual environments, test runners.
- Files: `pyproject.toml` (uv; deps numpy, pydantic, pyyaml, typer, fastapi, uvicorn; dev deps
  pytest, hypothesis, ruff, pre-commit), `src/neurodrive/__init__.py`, `src/neurodrive/cli.py`
  (stub `nd` command), `tests/test_smoke.py`, `README.md` check. The repository already exists:
  do **not** run `git init`, and keep the provided `.gitignore`.
- Verify: `uv sync`, `uv run pytest` (1 passing test), `uv run nd --help` lists commands, ruff
  clean, `git status` shows no ignored files being tracked.

**0.2 Seeded randomness module**
- Concept: pseudo-random generators, seeds, and why every random draw must be traceable.
- Files: `core/rng.py` (make a generator from a seed; spawn independent child generators for
  sub-components such as "init", "mutation", "start positions"), `tests/core/test_rng.py`.
- Verify: the same seed gives identical sequences; different child streams are independent and
  stable; a test confirms there are no global `np.random.*` calls in `src/` (simple source scan).

**0.3 Server skeleton**
- Concept: client/server, REST vs WebSocket (request/response vs a continuous stream).
- Files: `server/app.py` (health endpoint, `/ws/live` echo), `server/schemas.py` (hello message),
  `cli.py` (`nd serve`), `tests/server/test_health.py`. As built: also `server/__init__.py`, and
  the dependencies `websockets` (runtime; Uvicorn needs it to serve WebSockets) and `httpx2`
  (dev; FastAPI's test client).
- Verify: test client gets health OK; the WebSocket echo test passes.

**0.4 Web skeleton with three panels**
- Concept: how the UI is split into views that share one selection store.
- Files: `web/` (Vite + React + TS), `web/src/App.tsx` (three resizable panels: Environment,
  Network, Experiment), `web/src/state/selection.ts` (empty Zustand store with the fields in
  ARCHITECTURE §8.1), `web/src/api/ws.ts` (connect + status), type generation script
  `npm run gen:types`, Playwright smoke test. As built: the generator is `web/scripts/gen-types.mjs`
  plus `scripts/export_schemas.py` (Python side), the models are listed in `WIRE_MODELS` in
  `server/schemas.py`, and `vite.config.ts` proxies `/api` and `/ws` to the server (ADR-015).
- Verify: `npm run dev` shows three panels plus a green "connected" badge when `nd serve` runs;
  `npm run e2e` passes.

**0.4b Views as pages, with a companion drawer and a layout switch** (added 2026-10-12, ADR-017)
- Concept: navigation (one page per view, each with its own address) versus seeing two things at
  once; focus + context as a way to keep linked views visible without crowding the screen.
- Files: `web/src/App.tsx` (page shell), a small router (`web/src/nav/`) mapping `#/environment`,
  `#/network`, `#/experiment` to pages, a top bar with view tabs and a layout switch
  (Page / Split / All), a collapsible companion drawer shell on each page (placeholder content
  until 1.8, 1.9 and 2.7 fill it), the layout choice remembered in the browser,
  `web/e2e/smoke.spec.ts` updated, Vitest tests for the routing and layout logic.
- Verify: the app opens on the Environment page; tabs and the browser back button switch pages;
  the drawer opens and closes; Split shows two chosen views and All shows the three panels; the
  connection badge and the selection store survive page changes; `npm run test` and
  `npm run e2e` pass.

**0.5 Docs and conventions check**
- Concept: decision records, checklists, learning journal.
- Files: confirm `docs/*` and `CLAUDE.md` commands match reality; fix any mismatch.
- Verify: every command in CLAUDE.md §6 that exists so far runs.

**0.6 Version control safety net and first phase release**
- Concept: secret scanning, pre-commit hooks, continuous integration, branches and tags as versions.
- Files: `.pre-commit-config.yaml` (gitleaks, detect-private-key, check-added-large-files at
  1000 KB, end-of-file and trailing-whitespace fixers, ruff; versions pinned with
  `pre-commit autoupdate`), `.github/workflows/ci.yml` (Python job: `uv sync`, ruff, pytest; web
  job: `npm ci`, build, Vitest; secret-scan job: gitleaks over the full history), `README.md`
  (setup + contribution rules).
- Decided 2026-10-12: Windows Smart App Control on the development machine blocks the generated
  `pre-commit.exe` launcher, and it stays on. Document `uv run python -m pre_commit ...` for manual
  runs, install the git hooks normally and verify they actually run; if a hook's own binary (for
  example gitleaks) is blocked, run that check in CI instead and say so in the README.
- Verify: `uv run python -m pre_commit run --all-files` passes; a deliberately planted fake key in a scratch
  file is blocked by the hook (then deleted, never committed); CI is green on GitHub; the user has
  enabled secret scanning and push protection in the repository settings where available.
- Release: run `/finish-phase 0` → merge into `main`, tag `phase-0`, push after confirmation.

---

## Phase 1 — Milestone 1: "One neuron can drive"

**Goal:** a car on a simple loop, steered by one hand-wired neuron whose weights the user drags
live, with the Environment view and Neuron view linked.
**Learn:** weighted sum, bias, activation functions, output ranges, sensors as numbers,
normalisation, "weights are behaviour".
**Prereqs:** multiplication and sums; the shape of tanh (taught).
**Depends on:** Phase 0. **Enables:** all later phases (environment, views, linking).
**Phase done when:** a hand-wired neuron completes a lap; the sigmoid failure is explained in the
journal; clicking the car highlights its neuron and editing a weight changes the driving live.
**Risks:** physics tuning; raycast edge cases (rays parallel to walls, corners).

**1.1 Activations and the single neuron**
- Concept: neuron = weighted sum + bias, passed through an activation function; output ranges of
  tanh (−1..1), sigmoid (0..1), ReLU (0..∞), linear.
- Files: `core/activations.py`, `core/neuron.py`, `tests/core/test_activations.py`, `tests/core/test_neuron.py`.
- Verify: hand-computed examples (for instance inputs [2, 1], weights [0.5, −1], bias 0.2 →
  sum 0.2 → tanh ≈ 0.1974) match to 1e-9.

**1.2 Track model**
- Concept: representing a track as a centreline plus width; arc length; tangents; offset curves.
- Files: `env/track.py`, `tracks/train_loop_01.json` (a smooth loop with gentle and medium curves),
  `scripts/make_tracks.py`, `tests/env/test_track.py`.
- Verify: length is plausible; boundaries do not self-intersect; projection of points on the
  centreline returns e ≈ 0; projection of a point 3 m to the left returns e ≈ +3.

**1.3 Car physics (batched)**
- Concept: kinematic bicycle model, fixed time step, semi-implicit Euler integration.
- Files: `env/car.py`, `tests/env/test_car.py`.
- Verify: full throttle on a straight approaches v_max and never exceeds it; positive steer turns
  right (heading decreases); zero throttle slows down through drag; dead cars do not move;
  batched N = 3 equals three separate N = 1 runs; at high speed the car cannot turn tighter than
  the grip limit allows (turn radius ≥ v² / a_lat_max), so a hard steer at speed makes it run wide.

**1.4 Raycast sensors**
- Concept: ray–segment intersection; why sensor readings are normalised to [0, 1].
- Files: `env/sensors.py`, `tests/env/test_sensors.py` (including a slow, simple scalar reference
  implementation used only in tests).
- Verify: hand cases (a wall 10 m straight ahead → centre ray reads 10/50 = 0.2); a ray with no
  hit reads 1.0; batched results equal the scalar reference on random poses (Hypothesis).

**1.5 Progress, crash and termination**
- Concept: measuring progress along the track; why progress must only reward forward motion.
- Files: `env/track.py` (windowed projection), `env/world.py` (World: reset, step, termination),
  `tests/env/test_world.py`.
- Verify: driving forward around the loop increases P by ≈ L_track per lap; driving backwards
  reduces P; leaving the road ends the episode; stall detection ends a stopped car; determinism
  (same seed and actions → identical trajectory).

**1.6 Policies, episode runner and recording**
- Concept: the policy as a function from observations to actions; recording an episode for replay.
- Files: `env/policies.py` (Policy protocol, KeyboardPolicy, ConstantPolicy), `env/episode.py`
  (run episode, EpisodeRecording v1), `env/sensors.py` (observation builder v1),
  `tests/env/test_episode.py`.
- Verify: a recording saved and reloaded is identical; observation values are in the documented
  ranges; replaying recorded actions reproduces the same trajectory.

**1.7 Live session over WebSocket**
- Concept: a simulation loop running on the server and streaming frames.
- Files: `server/live.py`, `server/schemas.py` (scenario, frame, commands), `server/app.py`,
  `tests/server/test_live.py`.
- Verify: test client loads the scenario, receives frames, pause/step/reset behave as described;
  frame rate ≤ 30 per second.

**1.8 Environment view v1**
- Concept: rendering world coordinates on screen (flipping y, camera follow, zoom).
- Files: `web/src/views/environment/*` (track, car, rays with distance labels, HUD with speed and
  actions, play/pause/step/speed controls, keyboard driving), `web/src/api/*`.
- Verify: drive a lap manually with the keyboard; rays shorten as walls approach; crashing ends
  the episode and shows the reason; Playwright smoke test is updated.

**1.9 Neuron policy and Neuron view**
- Concept: steer = tanh(w_L·d_left + w_R·d_right + b), with a fixed throttle; the neuron as a
  tiny controller.
- Files: `env/policies.py` (NeuronPolicy; inputs chosen from the observation), server `set_param`
  command, `web/src/views/network/NeuronView.tsx` (inputs, weight sliders, weighted sum, activation
  curve with a live dot, output → steering arrow), selection linking (click car ↔ neuron).
- Verify: dragging a weight changes steering in the next frame; switching the activation to
  sigmoid is possible from the UI; unit test for NeuronPolicy maths.

**1.10 Milestone 1 experiments**
- Concept: hand-tuning as "the human is the learning algorithm"; the sigmoid range failure.
- Work: find weights that complete a lap; flip the signs; swap tanh for sigmoid; save the
  hand-wired controller as `configs/baselines/hand_neuron.yaml` for later exams.
- Verify: a lap completed and recorded; journal entry explains why sigmoid only ever steers one way
  (its output is always positive) and why the weights need opposite signs.

---

## Phase 2 — Multilayer network and Network view

**Goal:** real MLP policies driving many cars; a live Network view; the first version of the
output-neuron inspector and decision trails.
**Learn:** layers, the matrix form of a layer, forward propagation, parameter counting,
initialisation and symmetry, why hidden layers need nonlinearity.
**Depends on:** Phase 1. **Enables:** Phases 3, 5, 6, 7.
**Phase done when:** forward pass matches a hand-computed 2-2-1 example; clicking any of 20 random
cars shows its live network; clicking an output explains it; trail scrubbing works.
**Risks:** SVG performance (fine at this size); keeping frames small.

**2.1 Network spec and parameters**
- Concept: layer sizes, weight matrices and bias vectors; counting parameters; the flat "genome".
- Files: `core/nn/spec.py`, `core/nn/network.py`, `core/nn/io.py`, `tests/core/nn/test_network.py`.
- Verify: [7,16,16,2] has 434 parameters; flatten then unflatten gives back the same network;
  save/load round-trip is exact.

**2.2 Forward pass with trace**
- Concept: forward propagation layer by layer; why we keep every intermediate value.
- Files: `core/nn/forward.py`, `tests/core/nn/test_forward.py`.
- Verify: hand-computed 2-2-1 example matches to 1e-9; a batch of observations equals row-by-row
  forward passes.

**2.3 Population forward pass**
- Concept: evaluating N different networks at once (stacked weights).
- Files: `core/nn/population.py`, `tests/core/nn/test_population.py`.
- Verify: equals a loop of single forward passes for N = 20.

**2.4 Initialisation**
- Concept: why weights start random; scale matters (uniform vs Xavier); symmetry with zero init.
- Files: `core/nn/init.py`, `tests/core/nn/test_init.py`, `scripts/zero_init_demo.py`.
- Verify: Xavier variance matches the formula statistically; the zero-init demo shows all hidden
  neurons identical.

**2.5 Freeze the observation/action contract v1**
- Concept: why every method must see the same inputs and act through the same outputs.
- Files: `env/sensors.py` (finalise), `tests/env/test_contract.py` (shapes, ranges, order),
  `docs/ARCHITECTURE.md` §6.4–6.5 confirmed.
- Verify: contract test passes; ADR-005 marked accepted.

**2.6 MLP and Population policies, multi-car scenarios**
- Concept: one brain per car; random brains behave very differently.
- Files: `env/policies.py` (MLPPolicy, PopulationPolicy with traces), `server/live.py` (scenario
  "20 random networks", seed selectable), tests.
- Verify: 20 cars with 20 seeded networks; the same seed reproduces the same behaviours.

**2.7 Network view v1**
- Concept: reading a network diagram (edge colour = sign, thickness = |weight|, node fill = activation).
- Files: `web/src/views/network/NetworkDiagram.tsx`, legend, hover tooltips (exact values).
- Verify: the selected car's activations update live; the input nodes show the same numbers as the
  Environment view's rays.

**2.8 Selection linking**
- Concept: one shared selection drives every view.
- Files: `web/src/state/selection.ts`, click handlers in both views, `select_car` command.
- Verify: Vitest for store transitions; Playwright: clicking a car switches the diagram.

**2.9 Explain: contributions and sensitivity (finite differences)**
- Concept: contribution c_i = w_i·a_i; sensitivity = how much an output moves when one input is
  nudged slightly.
- Files: `explain/contributions.py`, `explain/sensitivity.py`, `tests/explain/*`.
- Verify: contributions sum to z_j − b_j exactly; sensitivity on a linear network equals its
  weights.

**2.10 Output inspector v1 and what-if**
- Concept: explaining one decision.
- Files: `server/app.py` (`explain/forward`), `web/src/views/network/OutputInspector.tsx`,
  `WhatIfPanel.tsx`, ray colouring by sensitivity in the Environment view.
- Verify: on pause, dragging the left-ray slider changes steering as the sensitivity predicts
  (sign and rough size).

**2.11 Decision trail and scrubbing**
- Concept: decisions over time; linking a moment to the network state.
- Files: `web/src/views/environment/Trail.tsx`, playback clock in `web/src/state/`, the live
  session keeps a rolling recording of the selected car.
- Verify: trail coloured by the selected output; clicking a trail point shows the trace at that step.

**2.12 Study S2a: random brains**
- Work: inspect 20 random networks; explain one circling car and one crashing car using the
  views. Journal entry.

---

## Phase 3 — Loss, derivatives and backpropagation (toy lab)

**Goal:** a verified backprop implementation, trained on toy problems, with gradients visible.
**Learn:** loss functions, derivatives as sensitivity, the chain rule, backpropagation, gradient
checking, vanishing gradients.
**Prereqs:** basic derivatives (taught with numbers).
**Depends on:** Phase 2. **Enables:** Phases 4, 6, 9A, 10.
**Phase done when:** gradient check passes on random networks for every activation; XOR is solved;
the user can do backprop on a 2-2-1 network on paper.
**Risks:** sign and index errors (caught by the gradient check).

**3.1 Loss functions:** MSE and Huber plus their derivatives. Files: `core/losses.py`, tests.
Verify: hand examples; derivative matches finite differences.
**3.2 Activation derivatives:** Files: `core/activations.py`, tests. Verify: against finite
differences at many points; sigmoid's maximum slope is 0.25.
**3.3 Backprop for one neuron:** hand-worked example in the Brief first, then
`core/nn/backward.py` (single-neuron path), tests. Verify: matches the hand example.
**3.4 Backprop for the full MLP (batched):** gradients for all W and b plus the gradient with
respect to the inputs. Files: `core/nn/backward.py`, tests. Verify: shapes, and the 2-2-1 hand example.
**3.5 Gradient checker:** Files: `core/nn/gradcheck.py`, `tests/core/nn/test_gradcheck.py`.
Verify: relative error < 1e-6 across activations, architectures and seeds. *The most important
test in the project.*
**3.6 Sensitivity via backprop:** switch `explain/sensitivity.py` to input gradients and keep
finite differences as a test oracle. Verify: they agree within 1e-5.
**3.7 Toy tasks and plain gradient descent:** `learn/supervised/toy_tasks.py` (line, sine, XOR,
two-spirals), `learn/supervised/train_toy.py`; the Learner contract v0 writes a run folder
(`experiment/store.py` v0). Verify: line fit recovers slope and intercept; XOR loss < 0.01.
**3.8 Toy lab in the Experiment view:** `web/src/views/experiment/ToyLab.tsx` (loss curve,
prediction plot), edge glow by gradient size in the Network view, `GET runs` endpoints.
Verify: open a toy run and scrub epochs; the predictions improve visibly.
**3.9 Study S2b:** sigmoid vs tanh vs ReLU on sine and XOR (vanishing gradients). Journal.

---

## Phase 4 — Optimisers and experiment basics

**Goal:** the optimizer family, configs, CLI, multi-seed sweeps, run comparison.
**Learn:** minibatches, epochs, learning rate, momentum, RMSProp, Adam, train/validation split, overfitting.
**Depends on:** Phase 3. **Enables:** Phases 5, 6, 9A, 10.
**Phase done when:** every optimizer passes its step test; sweeps across 5 seeds show median and
IQR bands; the user can predict too-high learning-rate behaviour.

**4.1 Minibatching and splits:** seeded shuffling, minibatch iterator, train/validation split.
Files: `learn/supervised/datasets.py`, tests. Verify: every example appears once per epoch; the
same seed gives the same order.
**4.2 Optimizer contract and SGD:** `core/optim/base.py`, `core/optim/sgd.py`, tests. Verify: one
step on f(w) = (w − 3)² from w = 0 with lr 0.1 gives w = 0.6.
**4.3 Momentum:** `core/optim/momentum.py`, tests. Verify: first two steps match hand values.
**4.4 RMSProp:** `core/optim/rmsprop.py`, tests. Verify: hand values.
**4.5 Adam:** `core/optim/adam.py`, tests, including bias correction. Verify: hand values for 3 steps.
**4.6 Config system and `nd train`:** `experiment/config.py` (Pydantic), `configs/toy_*.yaml`,
`cli.py`. Verify: invalid configs give clear errors; a toy run is created from YAML.
**4.7 Sweeps and overlays:** `nd sweep`, `experiment/stats.py` (median, IQR, bootstrap CI),
`web/src/views/experiment/RunList.tsx` + `Charts.tsx` (overlaid curves with bands). Verify: stats
unit tests; 5-seed sweep displayed.
**4.8 Study S3:** learning-rate sweep; optimizer comparison on sine across 5 seeds; overfitting
demo with a validation curve. Study report.
**4.9 (opt) Reference cross-check:** `reference/torch_check.py` compares our gradients and Adam
steps with PyTorch.

---

## Phase 5 — The Driving Exam, baselines and training infrastructure

**Goal:** a fair, deterministic yardstick; baselines; background training; the run timeline.
**Learn:** experimental design: controls, held-out tests, baselines, budgets.
**Depends on:** Phases 2 and 4. **Enables:** Phases 6–12.
**Phase done when:** baselines have exam cards; random search trains in the background; the
timeline replays any checkpoint with the network view linked.

**5.1 Track set:** `scripts/make_tracks.py`, `tracks/` with 1 training track and 3 held-out tracks
(tighter hairpins, a long sweeper, the mirrored training track). Verify: all pass the track tests;
the expert completes each.
**5.2 Driving metrics:** `experiment/metrics.py` (definitions in EXPERIMENT_PROTOCOL §3), tests
on synthetic recordings.
**5.3 Scripted expert:** `env/experts.py` (pure pursuit + curvature speed control), tests.
Verify: completes 2 laps on every track without crashing.
**5.4 Driving Exam v1:** `experiment/exam.py`, `nd exam`. Verify: deterministic results; the exam
of a recorded policy matches its recording.
**5.5 Learner contract v1 and Runner:** `learn/base.py`, `experiment/runner.py`,
`experiment/store.py` v1 (run folder v1, budgets, checkpoint and exam schedules, resume).
Verify: a dummy learner produces a valid run folder; killing and resuming continues identically.
**5.5b SQLite mirror (ADR-007):** `experiment/db.py`, `cli.py` (`nd db rebuild`, `nd db check`),
`tests/experiment/test_db.py`. Concept: relational tables, and why one store is the master copy and
the other a mirror. Verify: a database rebuilt from folders equals the incremental one; deleting
and rebuilding loses nothing; one example query (median exam score across seeds) gives the same
answer as computing it from the folders.
**5.6 Baselines:** random network, hand-wired neuron, scripted expert — each examined and stored
as a baseline run. Verify: exam cards exist.
**5.7 Random search learner:** `learn/evolution/random_search.py`, `configs/random_search.yaml`.
Concept: the simplest gradient-free learner (try, keep the best). Verify: best-so-far fitness never decreases.
**5.8 Run launching and progress streaming:** `server/runs_watch.py`, `POST runs`, `/ws/runs`,
process spawning. Verify: launching from the UI creates a run; charts grow live.
**5.9 Replay endpoint and run timeline:** `POST replay`, `checkpoints/{unit}/diff`,
`web/src/views/experiment/Timeline.tsx`, Network-view diff colouring. Verify: scrubbing changes all
three views consistently (Playwright test).
**5.10 Study S4:** random search vs baselines (5 seeds). Study report.

---

## Phase 6 — Supervised imitation (backprop meets driving)

**Goal:** a network that learns to drive from demonstrations, and a documented understanding of
why imitation fails and how DAgger helps.
**Learn:** behavioural cloning, datasets from demonstrations, imbalance, why low loss ≠ good
driving, distribution shift (compounding errors), DAgger.
**Depends on:** Phases 4 and 5. **Enables:** Phases 8 and 11.
**Phase done when:** the cloned car laps the training track; the distribution-shift failure is
documented with evidence; the DAgger improvement is measured on held-out tracks.
**Risks:** noisy human labels; most frames being "go straight".

**6.1 Demonstration recorder:** `learn/supervised/demos.py`, `nd demo-record` (expert or
keyboard via the live session), stored datasets with metadata. Verify: dataset shapes; frames
counted as demonstration budget.
**6.2 Dataset inspector:** `web/src/views/experiment/DatasetView.tsx` (action histograms, the
observations covered on a track map). Verify: displays the expert dataset.
**6.3 Imitation learner:** `learn/supervised/imitation.py`, `configs/imitation.yaml`
(feedback_kind = labels); an exam every epoch. Verify: overfits a 10-sample set; the training run
completes with exams.
**6.4 Study S5a — loss vs driving:** plot validation loss against exam progress per epoch; find
where they disagree.
**6.5 Study S5b — failure study:** perturbed starts (lateral offset, heading offset); find a crash
in a state the expert never visited; inspect it in the Network view.
**6.6 DAgger:** `learn/supervised/dagger.py`; the learner drives, the expert labels the visited
states, the data is aggregated and retrained. Verify: teacher queries are counted in the budget;
robustness improves.
**6.7 Study S5c:** human teacher vs scripted teacher; DAgger vs plain cloning on held-out tracks.

---

## Phase 7 — Genetic algorithm (fixed topology)

**Goal:** reproduce the reference video (a population evolving to fast laps) with full inspection.
**Learn:** genome, fitness, selection (tournament, rank), mutation (Gaussian; rate and strength),
crossover (uniform; why it is awkward for networks), elitism, diversity, exploration vs
exploitation, fitness exploits.
**Depends on:** Phase 5. **Enables:** Phases 8, 9, 11.
**Phase done when:** the training track is mastered across ≥ 5 seeds; sweeps are done; at least one
fitness exploit is documented.

**7.0 Performance check:** profile one generation (50 cars × 60 s). If it exceeds the target in
ARCHITECTURE §9, propose optimisations (⚑ if Numba is needed).
**7.1 Genome ↔ network:** uses `core/nn/network.py` flatten/unflatten; tests.
**7.2 Fitness v1 and evaluation:** `learn/evolution/ga.py` (evaluate a population in one batched
World). Verify: fitness ranking matches episode summaries.
**7.3 Selection operators:** tournament and rank. Tests on fixed fitness arrays (expected counts).
**7.4 Mutation and crossover:** tests (mutation statistics; crossover takes each gene from one parent).
**7.5 Elitism and the generation loop:** the GA learner (feedback_kind = episode_score),
best-of-generation checkpoints, population stats in `extra/`. Verify: with elitism, best fitness
never decreases on a fixed evaluation; determinism.
**7.6 Population mode in the Environment view:** the whole generation races, coloured by rank;
the previous best shown as a ghost; click any car to see its network.
**7.7 Evolution charts:** best/mean/worst fitness, fitness distribution, diversity (mean pairwise
weight distance).
**7.8 Lineage and mutation diff:** `extra/lineage.jsonl`; the champion's ancestry; the Network
view highlights the weights mutated since the parent.
**7.9 Study S6:** population size and mutation sweeps; fitness design ablations (progress only,
no time bonus, ...); GA vs random search at equal env steps; held-out tracks. Study report.

---

## Phase 8 — First comparative study: imitation vs evolution

**Goal:** the first evidence-based answer to the central question.
**Learn:** fair budgets in several currencies, statistics with few seeds, confounders.
**Depends on:** Phases 6 and 7.
**Phase done when:** study report S7 is written, with charts, and every claim points to data.

**8.1 Comparison page:** `experiment/compare.py`, `POST compare`,
`web/src/views/experiment/Compare.tsx` (exam metrics side by side; curves against env steps;
native metrics in separate labelled panels).
**8.2 Budget accounting review:** check that every learner reports env steps, demonstration frames,
teacher queries and wall-clock time; add a budget table to comparisons.
**8.3 Study S7:** same architecture, 10 seeds each: imitation, DAgger, GA, random search.
Training track + held-out tracks.
**8.4 Write-up:** `docs/studies/S7.md`, including what each method's feedback made easy or hard.

---

## Phase 9 — Advanced evolution

**9A Evolution Strategies** (bridge between evolution and gradients)
- Learn: estimating a gradient from fitness alone (perturb, score, average); antithetic sampling;
  fitness shaping; using Adam with an estimated gradient.
- Tasks: 9A.1 ES learner `learn/evolution/es.py` (tests on a quadratic: the estimated gradient
  points the right way); 9A.2 configs and runs; 9A.3 Study S8a: ES vs GA at equal budget.

**9B NEAT** (evolving structure) — time-boxed
- Learn: genomes that describe structure, innovation numbers, speciation, complexification.
- Tasks: 9B.1 genome and innovation tracking; 9B.2 a forward pass for arbitrary DAGs (with
  ForwardTrace); 9B.3 mutation (add node, add connection, weight) and crossover by innovation
  number; 9B.4 speciation and fitness sharing; 9B.5 Network view layout for arbitrary DAGs;
  9B.6 Study S8b: fixed vs evolving topology, including network size.
- Each task is verified with unit tests on small hand-built genomes.
- Risk: NEAT is the largest single algorithm. Split further if needed.

---

## Phase 10 — Reinforcement learning

**Goal:** learning from per-step rewards through interaction.
**Learn:** MDP (state, action, reward, transition), return, discount, stochastic policies,
log-probability, REINFORCE, variance and baselines, the value function, advantage (GAE), PPO
clipping, entropy bonus, reward design.
**Depends on:** Phases 3, 4, 5 (and 7 for comparison).
**Phase done when:** PPO laps the training track in ≥ 3 of 5 seeds; Study S9 is written.
**Risks:** the hardest phase (instability, sample hunger). Toy problem first; small networks;
many cars in parallel in one batched World.

**10.1 Reward v1 review and step-reward tap:** confirm the `env/feedback.py` reward; ADR if changed.
**10.2 Returns and discounting:** `learn/rl/returns.py`, tests on hand sequences.
**10.3 Gaussian policy:** `learn/rl/gaussian_policy.py` (mean from the network, learnable log
standard deviation, sampling, log-probability and its gradient); gradient check.
**10.4 Toy environment:** a 1D "stay in the centre of the corridor" task in `learn/rl/toy_env.py`.
**10.5 REINFORCE:** `learn/rl/reinforce.py` on the toy task, then on the track.
**10.6 Baseline and critic:** a value network; advantage = return − value; `learn/rl/actor_critic.py`.
**10.7 GAE and PPO:** `learn/rl/ppo.py` (clipped objective, several epochs per batch, entropy bonus).
**10.8 RL views:** a reward trace along the trail, a value estimate overlay, the policy's standard
deviation over training.
**10.9 Study S9:** reward shaping, discount and entropy ablations; RL vs GA vs imitation under the
protocol.

---

## Phase 11 — Hybrids

**Goal:** test whether combining feedback types beats each alone.
**Depends on:** Phases 6, 7, 10.
**11.1 Distillation:** the GA champion becomes the teacher for imitation (`learn/hybrid/distill.py`).
**11.2 Pretrain then fine-tune:** imitation → GA, and imitation → PPO
(`learn/hybrid/pretrain_finetune.py`).
**11.3 Evolved initialisation → gradient refinement.**
**11.4 Study S10.**

---

## Phase 12 — Final study and polish

**12.1 Generalisation study S11:** all methods on all held-out tracks plus procedurally generated
tracks.
**12.2 Robustness:** sensor noise, changed car physics, unseen starts.
**12.3 Performance pass:** profile and optimise against the targets.
**12.4 Guided tour mode:** a step-by-step walkthrough in the UI from the neuron to PPO.
**12.5 Final report:** `docs/studies/FINAL.md` answering the central question.
**12.6 Final validation:** full test suite, all studies reproducible from their configs and seeds.
