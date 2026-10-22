# NeuroDrive — Implementation Checklist

Legend: ☐ planned · ◐ in progress · ☑ implemented (not yet verified) · ✅ verified · ⏸ deferred
Update rules: Claude Code marks ◐ when starting, ☑ when the code is written, ✅ only after the
task's verification has passed. Add the date and a short note when verifying. When a phase is
released, add its tag (e.g. `phase-1`) to the phase's header row.

| ID | Task | Status | Verified on | Notes |
|---|---|---|---|---|
| — | Planning docs (CLAUDE.md, docs/*) | ✅ | 2026-10-10 | Python core + web viewer (ADR-001); revised ADR-003/004/007; ADR-014 |
| — | GitHub repo created, planning docs committed to `main` (user) | ✅ | 2026-10-10 | Commit `e7f2ce2` on `origin/main` |
| **Phase 0** | **Foundations** — tag `phase-0` | | | |
| 0.1 | Python project skeleton | ✅ | 2026-10-10 | uv + hatchling, Python 3.12; pytest 1 passed, `nd --help` lists `version`, ruff clean |
| 0.2 | Seeded randomness module | ✅ | 2026-10-10 | Name-keyed child streams; 17 tests incl. no-global-RNG source scan (planted violation caught); full suite 18 passed |
| 0.3 | Server skeleton | ✅ | 2026-10-11 | `/api/health` + `/ws/live` hello and echo; `nd serve` checked against a real server; 21 tests passed; added websockets, httpx2 (dev) |
| 0.4 | Web skeleton with three panels | ✅ | 2026-10-12 | Vite 8 + React 19 + TS; Vitest 3 passed; `npm run build` clean; Playwright 2 passed; kill/restart reconnect checked; ADR-015 (dev proxy) |
| 0.4b | Views as pages + companion drawer + layout switch | ✅ | 2026-10-12 | ADR-017; hash routing, layout store; Vitest 14 passed; Playwright 6 passed (incl. no reload / no reconnect on page change); screenshots light, dark, phone |
| 0.5 | Docs and conventions check | ✅ | 2026-10-12 | All runnable §6 commands run; 8 doc mismatches fixed + stale web lock file; open: pre-commit.exe blocked by Windows policy, `.env.example` vars unused, README stale (see journal) |
| 0.6 | Version control safety net + release `phase-0` | ✅ | 2026-10-12 | ADR-014; planted secret blocked; CI green on GitHub (all 4 jobs, first run); secret scanning + push protection enabled by user |
| **Phase 1** | **Milestone 1: One neuron can drive** — tag `phase-1` | | | |
| 1.1 | Activations and the single neuron | ✅ | 2026-10-13 | `uv run pytest` 44 passed (23 new); hand values (tanh(0.2)=0.19738) match to 1e-9; sigmoid safe at ±1000; ruff clean |
| 1.2 | Track model | ✅ | 2026-10-13 | `uv run pytest` 70 passed (26 new in tests/env); circle: L=314.14 vs 314.16, e=+3.00 left / −3.00 right; train_loop_01 L=731 m, min bend radius 22.5 m, edges do not cross |
| 1.3 | Car physics (batched) | ✅ | 2026-10-15 | `uv run pytest` 87 passed (17 new in tests/env/test_car.py); hand step v=10.2333, psi=−0.029316; v_max reached in 4.7 s; batch N=3 bit-identical to 3 single runs; grip limit holds at all speeds; §6.3 values kept unchanged |
| 1.4 | Raycast sensors | ✅ | 2026-10-15 | `uv run pytest` 105 passed (18 new in tests/env/test_sensors.py); wall 10 m ahead → 0.2; no hit → 1.0; Hypothesis: batched = scalar reference to 1e-9 (60 track + 150 random-wall cases); two planted bugs were caught; ~21 ms for 100 cars |
| 1.5 | Progress, crash, termination | ✅ | 2026-10-15 | `uv run pytest` 128 passed (23 new in tests/env/test_world.py); scripted driver laps train_loop_01: P=731.02 vs L=730.74, FINISHED at 76 s; backwards P<0; crash at \|e\|>5 m; stall at exactly 60 steps; wrap bug planted → 3 tests failed; batch = single-car worlds; follow-up 2026-10-15: defaults now 2 laps / 90 s, seeded random starts in `reset` (135 tests passed) |
| 1.6 | Policies, episode runner, recording | ✅ | 2026-10-15 | `uv run pytest` 156 passed (21 new in tests/env/test_episode.py); save → load identical; replay of 4-car random-start run identical bit for bit (also after save/load); obs o1–o6 in [0,1], o7 in (−1,1]; touched world.py (`start`, `ready`) and §6.11 note |
| 1.7 | Live session over WebSocket | ✅ | 2026-10-15 | `uv run pytest` 183 passed (27 new in tests/server/test_live.py; echo test removed); fake-clock runs at speeds 0.5/1/2/8: ≤30 frames/s, 30×speed steps/s; real `nd serve` check: 60 frames and 60 steps in 2 s at speed 1; types regenerated, Vitest 14, build, Playwright 6 pass; ADR-018, api v1.2; two planted bugs caught |
| 1.8 | Environment view v1 | ✅ | 2026-10-16 | Vitest 41 passed (27 new: geometry, keys, status words, live store); `tsc` + `npm run build` clean; Playwright 12 passed (6 new: track drawn, play/pause/step, arrow keys, crash reason + reset, compact drawer view, session kept across pages); screenshots light, dark, phone; manual lap by the user still to do; follow-up 2026-10-16: UI controls for driver, cars, random start + seed, selected car (Vitest 44, Playwright 15 passed) |
| 1.9 | Neuron policy and Neuron view | ✅ | 2026-10-16 | `uv run pytest` 217 passed (22 NeuronPolicy + 13 server tests new); worked example z=1.3 → tanh 0.8617, z=−1.1 → −0.8005, sigmoid 0.7858 / 0.2497 matches; Vitest 54, Playwright 25 (10 new), tsc + build clean; hand weights ±8 drive 90 s without crashing, zeros crash at 9.8 s, signs flipped crash; ADR-019, api v1.3; a planted wrong-rays bug failed 8 tests; follow-up 2026-10-18: neuron settings last for the whole session (ADR-020), restored after a reconnect (pytest 43 server, Vitest 57, Playwright 26) |
| 1.10 | Milestone 1 experiments (S1) | ✅ | 2026-10-16 | Study S1 (`docs/studies/S1.md`, `scripts/study_s1.py`): weights −12/+12, tanh, throttle 0.24 finish 2 laps in 86.3 s (10/10 exam-like starts, 10/10 seeded random starts), lap recorded and replayed identically; window only throttle 0.23–0.25; flipped signs crash at 3.6 s; 108 sigmoid settings none finish, min steer never below 0; `configs/baselines/hand_neuron.yaml` + 5 guard tests |
| 1.11 | Half-size Monza, both directions | ✅ | 2026-10-18 | `monza_cw` (train) and `monza_ccw` (heldout): 2,900 m each, traced from the user's circuit map at half the real length; tightest bend 11.3 m (Roggia), T1 12.3 m; 6 new track tests; edge check now compares only nearby segments (load 2.4 s → 0.05 s, memory 280 → 32 MB); pytest 229, Vitest 58, Playwright 26 |
| 1.12 | Study S1b: neuron on Monza, both directions | ✅ | 2026-10-18 | Study S1b (`docs/studies/S1b.md`, `scripts/study_s1b.py`): per-track time limit 2 laps at 16 m/s (ADR-022; 363 s Monza, 92 s loop); no fixed throttle finishes on Monza either way (fastest safe 0.19 cw / 0.1825 ccw would need 392 s / 408 s); S1 baseline crashes in the first tight bend at 19 m/s vs a 12.5–15.7 m/s grip speed; sigmoid never steers left (best 215 m cw, 86 m ccw); `WallGrid` sensors bit-identical to the full search, 13× faster on Monza; pytest 234, Vitest 58, Playwright 26 |
| **Phase 2** | **Multilayer network and Network view** — tag `phase-2` | | | |
| 2.1 | Network spec and parameters | ✅ | 2026-10-19 | `uv run pytest` 266 passed (32 new in tests/core/nn/test_network.py); [7,16,16,2] = 128 + 272 + 34 = 434 parameters; flatten → unflatten identical bit for bit; hand [2,2,1] flattens to 1…9; `.npz` save → load exact; ruff clean; no contract change |
| 2.2 | Forward pass with trace | ✅ | 2026-10-19 | `uv run pytest` 281 passed (15 new in tests/core/nn/test_forward.py, 47 in tests/core/nn); hand 2-2-1: z1 = [−1.5, 1.5], a1 = ±0.905148, output −0.805148 to 1e-9; batch of 20 = row-by-row to 1e-9 for 3 specs; ruff clean; no contract change |
| 2.3 | Population forward pass | ✅ | 2026-10-19 | `uv run pytest` 297 passed (16 new in tests/core/nn/test_population.py, 63 in tests/core/nn); N = 20 equals a loop of single forward passes (all z and a) to 1e-9; hand example [[1,1],[2,3]] gives 3 and 1.5; from_genomes agrees with Network.unflatten; ruff clean; no contract change |
| 2.4 | Initialisation | ✅ | 2026-10-19 | `uv run pytest` 310 passed (13 new in tests/core/nn/test_init.py); Xavier weight variance within 3% of 2/(n_in+n_out) on 300×400 and 400×200 layers; same seed → identical network; `scripts/zero_init_demo.py`: spread across hidden neurons 0.0000 for zero and all-0.1 weights, 0.3635 / 0.2791 for Xavier; ruff clean; no contract change |
| 2.5 | Freeze observation/action contract v1 | ✅ | 2026-10-19 | `uv run pytest` 323 passed (13 new in tests/env/test_contract.py); count, order, ray angles, ranges (o1–o6 in [0,1], o7 in (−1,1]), hand values, batch = single, random-action driving, action clipping and signs pinned; three planted bugs (ray order, o7 scale, steer sign) each failed the contract test; `OBSERVATION_VERSION`/`OBSERVATION_NAMES` added; ADR-005 accepted; ruff clean |
| 2.6 | MLP / Population policies, multi-car | ✅ | 2026-10-19 | `uv run pytest` 356 passed (new: 23 in tests/env/test_mlp_policy.py, 8 in tests/server/test_live.py, 2 in tests/env/test_contract.py); 20 seeded networks: same seed → bit-identical recording, other seed differs; seed 0: 20 different first actions, 14 stall at the start (braking at speed 0), 6 crash, none finish; Vitest 58, tsc + build clean; two planted bugs (all cars use member 0, seed ignored) caught; `observe()` replaces the duplicated observation code (recordings still bit-identical), `ACTION_*` constants added; ADR-023, api v1.4; Playwright 27 passed (incl. the new `random networks: 20 cars` test; checked on 2026-10-19 after the server was restarted) |
| 2.7 | Network view v1 | ✅ | 2026-10-19 | `uv run pytest` 368 passed (12 new: 7 in tests/server/test_live.py, 5 in tests/env/test_mlp_policy.py); Vitest 72 (13 layout + 1 store test new); Playwright 35 (8 new in e2e/network.spec.ts): 41 circles and 400 lines, sign colours and thickness, input circles = the Environment view's rays (metres / 50), activations change while stepping, another car shows other weights, hover readout and tooltips, compact drawer; trace output recomputed from the message weights matches to 2e-3; two planted bugs (stale trace, swapped sign colours) caught; light and dark screenshots checked; ADR-024, api v1.5 |
| 2.8 | Selection linking | ✅ | 2026-10-19 | Vitest 87 passed (14 new store-transition tests in `state/linking.test.ts`, 1 new layout test); Playwright 41 passed (6 new in `e2e/linking.spec.ts`: clicking a car on the canvas switches the diagram, ‹ › stepper moves the Environment's car (wraps), ray input / weight highlights that ray, a selected neuron stays selected across cars, drawer cell ↔ full diagram, new scenario clears the selection); compact drawer redesigned (input bars, hidden-layer cell strips, output bars with words) at the user's request, light + dark screenshots checked; planted bug (selectCar never sends) failed 1 Vitest + 3 Playwright tests; `uv run pytest` 368 passed (no Python change); no API change |
| 2.9 | Explain: contributions + sensitivity (finite diff.) | ✅ | 2026-10-19 | `uv run pytest` 395 passed (27 new in tests/explain); contributions sum to z − b for every neuron: worst gap 8.9e-16 over 200 random networks (6,800 neurons, 40 % bit-identical; the rest differ only by summation order); hand 2-2-1: hidden c = [0.5, −2] → −1.5, output c = [−1.810297, 0.905148] + 0.1 = −0.805148; sensitivity of a linear network = its weights to 1.7e-12 (two layers: W2·W1); hand chain rule ∂out/∂x = [0.361413, −0.361413] matched to 1.8e-9; two planted bugs (one-sided difference, transposed weights) failed 4 and 12 tests; ruff clean; no UI change |
| 2.10 | Output inspector v1 + what-if | ✅ | 2026-10-19 | ADR-025 (option A, chosen by the user): `selected.explain` per frame + `what_if` command, api v1.6, REST `explain/forward` deferred to Phase 5; `uv run pytest` 403 passed (8 new); Vitest 95 (8 new); Playwright 45 (4 new): paused, left ray +0.1 moves steer with the sensitivity's sign and close to its prediction (car 5: s = +0.051, steer 0.492 → 0.496, predicted 0.497); inspector receipt/sensitivity, ray colours = sensitivity row, what-if dropped on the next step and refused while playing; planted bug (what-if answered with the real observation) caught by both the Python and the browser check; light + dark screenshots checked; follow-up 2026-10-19 (user request): what-if ghost car (2 s of held real vs what-if action from the Python car physics, drawn as paths + see-through car, legend gives the gap), `what_if` sends only the changed inputs, Network-page car drop-down, clicking a car in the drawer's track tested; ADR-026, api v1.7; pytest 409, Vitest 96, Playwright 49 (4 new in e2e/ghost.spec.ts); planted bug (ghost used the real action twice) failed 2 tests |
| 2.11 | Decision trail and scrubbing | ✅ | 2026-10-20 | ADR-027 (A1 look without rewinding + B1 last 60 s of every car, chosen by the user), api v1.8: `selected.trail` in frames, `inspect_step` → `moment`; `uv run pytest` 418 passed (9 new): trail = the car's real path and actions, a new car gets its whole trail, 2 points per frame at speed 2, every `inspect_step` equals the live frame's trace and explanation bit for bit, only 60 s kept, refused while playing / for steps not remembered, crashed car's trail stops at the crash; Vitest 112 (16 new: trail store, playback clock, trail colours, clicking); Playwright 53 (4 new in e2e/trail.spec.ts): trail values = the network's steer/throttle at each step, slider and clicked point show that step's exact inputs, outputs and sensitivities, Back to now, step / play end the look back; planted bug (moment used the current observation) failed 1 Python + 1 browser test; seed 1 car 12: left ray 1.1 m at step 310 yet steer −0.021 (sensitivity to it −0.004) before it crashes; light + dark screenshots checked; inspect_step 0.2 ms, 3.8 MB for 20 cars; follow-up 2026-10-20 (user request): the HUD shows the step being looked at (speed, status, action chosen there, step and time, progress, rays, "now is step N"), `moment` gains progress and status, ADR-028, api v1.9; pytest 418 (2 new checks in existing tests), Vitest 112, Playwright 54 (1 new; planted bug, HUD kept the live progress, failed it) |
| 2.12 | Study S2a: random brains | ✅ | 2026-10-20 | Study S2a (`docs/studies/S2a.md`, `scripts/study_s2a.py`, figures + 4 view screenshots): 2,000 brains (seeds 0–99 × 20, random starts, exam rules): 72 % stall, 28 % crash, 13 time out, 1 finishes (seed 10 car 11, 83.0 s); 956 of 962 non-movers brake at step 0 (first throttle positive 52 %: a coin flip); no brain circles (|steer| never > 0.9; full-lock circle 9.15 m does not fit on the 12 m road), so the roadmap's circling car is replaced by a self-stopper (seed 5 car 18); crashers' wiring signs match coin flips (28 vs 34.6 with all four right), long drivers 8/14; crash case seed 1 car 12 blind to the left wall (s = −0.004); straight-line breakdowns within 0.06 of the actual change; steering chatter traced to rays grazing wall tips (full search identical); guard test `tests/server/test_study_s2a.py` (4 tests: study cars = app cars step for step, cases hold) caught a planted wrong-stream bug; pytest 422; follow-up 2026-10-20 (user request): grid of 3 tracks (loop, monza_cw, monza_ccw) × 6 shapes ([7,2] to [7,64,64,2] and 4 hidden layers), 36,000 brains, 50 min, a second run reproduced it exactly: stall/crash shares the same on Monza, random brains finish Monza (seed 51 car 8, 355.9 s; 3 of 6,000), survivors are the same brains across tracks, wider first layers give gentler outputs (Xavier), still no circling (would-be circler seed 28 car 14 [7,16,2] stalls after 13°); Monza finisher case + screenshot; guard tests 6 |
| **Phase 3** | **Backpropagation (toy lab)** | | | |
| 3.1 | Loss functions | ☐ | | |
| 3.2 | Activation derivatives | ☐ | | |
| 3.3 | Backprop for one neuron | ☐ | | |
| 3.4 | Backprop for the full MLP | ☐ | | |
| 3.5 | Gradient checker | ☐ | | |
| 3.6 | Sensitivity via backprop | ☐ | | |
| 3.7 | Toy tasks + plain gradient descent | ☐ | | |
| 3.8 | Toy lab in Experiment view | ☐ | | |
| 3.9 | Study S2b: activations | ☐ | | |
| **Phase 4** | **Optimisers and experiment basics** | | | |
| 4.1 | Minibatching and splits | ☐ | | |
| 4.2 | Optimizer contract + SGD | ☐ | | |
| 4.3 | Momentum | ☐ | | |
| 4.4 | RMSProp | ☐ | | |
| 4.5 | Adam | ☐ | | |
| 4.6 | Config system + `nd train` | ☐ | | |
| 4.7 | Sweeps and overlays | ☐ | | |
| 4.8 | Study S3: optimizers | ☐ | | |
| 4.9 | (opt) PyTorch reference cross-check | ☐ | | |
| **Phase 5** | **Driving Exam and infrastructure** | | | |
| 5.1 | Track set | ☐ | | |
| 5.2 | Driving metrics | ☐ | | |
| 5.3 | Scripted expert | ☐ | | |
| 5.4 | Driving Exam v1 | ☐ | | |
| 5.5 | Learner contract v1 + Runner | ☐ | | |
| 5.5b | SQLite mirror | ☐ | | ADR-007 |
| 5.6 | Baselines | ☐ | | |
| 5.7 | Random search learner | ☐ | | |
| 5.8 | Run launching + progress streaming | ☐ | | |
| 5.9 | Replay endpoint + run timeline | ☐ | | |
| 5.10 | Study S4: random search vs baselines | ☐ | | |
| **Phase 6** | **Supervised imitation** | | | |
| 6.1 | Demonstration recorder | ☐ | | |
| 6.2 | Dataset inspector | ☐ | | |
| 6.3 | Imitation learner | ☐ | | |
| 6.4 | Study S5a: loss vs driving | ☐ | | |
| 6.5 | Study S5b: failure study | ☐ | | |
| 6.6 | DAgger | ☐ | | |
| 6.7 | Study S5c: teachers, DAgger | ☐ | | |
| **Phase 7** | **Genetic algorithm** | | | |
| 7.0 | Performance check | ☐ | | |
| 7.1 | Genome ↔ network | ☐ | | |
| 7.2 | Fitness v1 + population evaluation | ☐ | | |
| 7.3 | Selection operators | ☐ | | |
| 7.4 | Mutation and crossover | ☐ | | |
| 7.5 | Elitism + generation loop | ☐ | | |
| 7.6 | Population mode (Environment view) | ☐ | | |
| 7.7 | Evolution charts | ☐ | | |
| 7.8 | Lineage + mutation diff | ☐ | | |
| 7.9 | Study S6 | ☐ | | |
| **Phase 8** | **Imitation vs evolution** | | | |
| 8.1 | Comparison page | ☐ | | |
| 8.2 | Budget accounting review | ☐ | | |
| 8.3 | Study S7 runs | ☐ | | |
| 8.4 | Study S7 write-up | ☐ | | |
| **Phase 9** | **Advanced evolution** | | | |
| 9A.1 | ES learner | ☐ | | |
| 9A.2 | ES configs and runs | ☐ | | |
| 9A.3 | Study S8a: ES vs GA | ☐ | | |
| 9B.1 | NEAT genome + innovations | ☐ | | |
| 9B.2 | DAG forward pass | ☐ | | |
| 9B.3 | NEAT mutation + crossover | ☐ | | |
| 9B.4 | Speciation + fitness sharing | ☐ | | |
| 9B.5 | Network view for arbitrary DAGs | ☐ | | |
| 9B.6 | Study S8b | ☐ | | |
| **Phase 10** | **Reinforcement learning** | | | |
| 10.1 | Reward review + step-reward tap | ☐ | | |
| 10.2 | Returns and discounting | ☐ | | |
| 10.3 | Gaussian policy | ☐ | | |
| 10.4 | Toy corridor environment | ☐ | | |
| 10.5 | REINFORCE | ☐ | | |
| 10.6 | Baseline + critic | ☐ | | |
| 10.7 | GAE + PPO | ☐ | | |
| 10.8 | RL views | ☐ | | |
| 10.9 | Study S9 | ☐ | | |
| **Phase 11** | **Hybrids** | | | |
| 11.1 | Distillation | ☐ | | |
| 11.2 | Pretrain then fine-tune | ☐ | | |
| 11.3 | Evolved init → gradient refinement | ☐ | | |
| 11.4 | Study S10 | ☐ | | |
| **Phase 12** | **Final study and polish** | | | |
| 12.1 | Generalisation study S11 | ☐ | | |
| 12.2 | Robustness tests | ☐ | | |
| 12.3 | Performance pass | ☐ | | |
| 12.4 | Guided tour mode | ☐ | | |
| 12.5 | Final report | ☐ | | |
| 12.6 | Final validation | ☐ | | |
