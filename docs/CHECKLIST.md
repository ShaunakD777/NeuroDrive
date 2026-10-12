# NeuroDrive — Implementation Checklist

Legend: ☐ planned · ◐ in progress · ☑ implemented (not yet verified) · ✅ verified · ⏸ deferred
Update rules: Claude Code marks ◐ when starting, ☑ when the code is written, ✅ only after the
task's verification has passed. Add the date and a short note when verifying. When a phase is
released, add its tag (e.g. `phase-1`) to the phase's header row.

| ID | Task | Status | Verified on | Notes |
|---|---|---|---|---|
| — | Planning docs (CLAUDE.md, docs/*) | ✅ | 2026-10-10 | Python core + web viewer (ADR-001); revised ADR-003/004/007; ADR-014 |
| — | GitHub repo created, planning docs committed to `main` (user) | ✅ | 2026-10-10 | Commit `e7f2ce2` on `origin/main` |
| **Phase 0** | **Foundations** | | | |
| 0.1 | Python project skeleton | ✅ | 2026-10-10 | uv + hatchling, Python 3.12; pytest 1 passed, `nd --help` lists `version`, ruff clean |
| 0.2 | Seeded randomness module | ✅ | 2026-10-10 | Name-keyed child streams; 17 tests incl. no-global-RNG source scan (planted violation caught); full suite 18 passed |
| 0.3 | Server skeleton | ✅ | 2026-10-11 | `/api/health` + `/ws/live` hello and echo; `nd serve` checked against a real server; 21 tests passed; added websockets, httpx2 (dev) |
| 0.4 | Web skeleton with three panels | ✅ | 2026-10-12 | Vite 8 + React 19 + TS; Vitest 3 passed; `npm run build` clean; Playwright 2 passed; kill/restart reconnect checked; ADR-015 (dev proxy) |
| 0.4b | Views as pages + companion drawer + layout switch | ☐ | | ADR-017 (added 2026-10-12) |
| 0.5 | Docs and conventions check | ✅ | 2026-10-12 | All runnable §6 commands run; 8 doc mismatches fixed + stale web lock file; open: pre-commit.exe blocked by Windows policy, `.env.example` vars unused, README stale (see journal) |
| 0.6 | Version control safety net + release `phase-0` | ☐ | | ADR-014 |
| **Phase 1** | **Milestone 1: One neuron can drive** | | | |
| 1.1 | Activations and the single neuron | ☐ | | |
| 1.2 | Track model | ☐ | | |
| 1.3 | Car physics (batched) | ☐ | | |
| 1.4 | Raycast sensors | ☐ | | |
| 1.5 | Progress, crash, termination | ☐ | | |
| 1.6 | Policies, episode runner, recording | ☐ | | |
| 1.7 | Live session over WebSocket | ☐ | | |
| 1.8 | Environment view v1 | ☐ | | |
| 1.9 | Neuron policy and Neuron view | ☐ | | |
| 1.10 | Milestone 1 experiments (S1) | ☐ | | |
| **Phase 2** | **Multilayer network and Network view** | | | |
| 2.1 | Network spec and parameters | ☐ | | |
| 2.2 | Forward pass with trace | ☐ | | |
| 2.3 | Population forward pass | ☐ | | |
| 2.4 | Initialisation | ☐ | | |
| 2.5 | Freeze observation/action contract v1 | ☐ | | |
| 2.6 | MLP / Population policies, multi-car | ☐ | | |
| 2.7 | Network view v1 | ☐ | | |
| 2.8 | Selection linking | ☐ | | |
| 2.9 | Explain: contributions + sensitivity (finite diff.) | ☐ | | |
| 2.10 | Output inspector v1 + what-if | ☐ | | |
| 2.11 | Decision trail and scrubbing | ☐ | | |
| 2.12 | Study S2a: random brains | ☐ | | |
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
