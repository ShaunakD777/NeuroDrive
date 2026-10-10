# NeuroDrive — Experiment and Testing Protocol

## 1. Verifying components

| Component | Verification |
|---|---|
| Neuron, activations, forward pass | Hand-computed small examples, exact match (tolerance 1e-9) |
| Derivatives, backprop, policy log-probabilities | Gradient check vs. finite differences, relative error < 1e-6 |
| Optimisers | First steps on a known function vs. hand calculation |
| Track geometry, raycasts | Hand cases + property tests vs. a simple scalar reference |
| Physics | Invariants: speed bounds, steering direction, dead cars frozen, batched = unbatched |
| Environment | **Determinism:** same seed + actions → identical trajectory |
| Learners | Sanity checks: imitation overfits 10 samples; GA improves a trivial fitness; RL solves the toy corridor |
| Runner and store | Valid run folders; resume gives the same result as an uninterrupted run |
| Server | Endpoint and WebSocket tests with the FastAPI test client |
| Web | Vitest for selection logic; Playwright smoke test for linking |

## 2. Rules for fair comparison

1. **Write the question and hypothesis before running anything.** Change one variable at a time.
2. **Fixed contract:** observation v1, action v1, physics v1, and the same network architecture
   for all methods unless the architecture is the variable being studied.
3. **The Driving Exam is the only cross-method yardstick.** Loss, fitness and reward are shown
   only within their own method, in separately labelled panels. Never on a shared axis.
4. **Budgets in several currencies:** environment steps (main x-axis for learning curves),
   wall-clock time, demonstration frames, teacher queries. Report all of them.
5. **Seeds:** at least 5, and 10 for headline studies. Report the median and interquartile range
   (IQR). Never report a single best run as the result.
6. **Baselines are always shown:** random network, random search, hand-wired neuron, scripted expert.
7. **Held-out tracks** are never used for training, tuning or checkpoint selection.
8. **Checkpoint selection** uses the training track only (or a validation track), never the
   held-out tracks.
9. **Failure analysis is required:** every study inspects at least one failure in the linked views.
10. **Reproducibility:** each study lists its configs, seeds, run IDs and the git commit.

## 3. Driving Exam v1

- **Tracks:** `train_loop_01` (reported separately) + every track with `split: heldout`.
- **Starts per track:** 5 positions at 0 %, 20 %, 40 %, 60 % and 80 % of the track length,
  centred and aligned with the tangent, plus the same 5 with a +1.5 m lateral offset → 10 episodes
  per track.
- **Limits:** 90 s time limit; target 2 laps; stall rule as in ARCHITECTURE §6.6.
- **Policy mode:** deterministic (RL policies use their mean action).
- **Per-episode metrics:**
  - `progress_m`: cumulative forward progress
  - `progress_frac`: progress ÷ (2 × track length), capped at 1
  - `completed`: the target laps were reached
  - `crashed`: the episode ended by leaving the road
  - `lap_time_s`: time of each completed lap
  - `mean_speed`: m/s while alive
  - `smoothness`: mean absolute change in steering per step (lower is smoother)
  - `time_to_crash_s`: if crashed
- **Aggregates per track and overall:** completion rate, crash rate, median and IQR of
  `progress_frac`, median best lap time (completed episodes only), median smoothness.
- The exam spec version is stored with every result. Changing the exam requires an ADR and a new
  version, and old results are not compared with new ones.

## 4. Statistics

- Median and IQR across seeds for every reported metric.
- For "A vs B" claims: a bootstrap 95 % confidence interval for the difference in medians
  (10 000 resamples, seeded). If the interval includes 0, say "no clear difference".
- Learning curves: median line with an IQR band, x-axis = environment steps.
- Small seed counts are a known limitation and are stated in every report.

## 5. Planned studies (mapped to the central question)

| ID | Phase | Question |
|---|---|---|
| S1 | 1 | How far can a hand-wired single neuron go, and why does sigmoid fail as a steering output? |
| S2a | 2 | What do random networks do, and can the views explain why? |
| S2b | 3 | How does the activation choice affect gradient-based learning (vanishing gradients)? |
| S3 | 4 | How do learning rate and optimizer choice affect convergence and stability? |
| S4 | 5 | Does random search beat the baselines, and at what budget? |
| S5a–c | 6 | Does low imitation loss mean good driving? Where does cloning fail, and does DAgger fix it? Human vs scripted teacher? |
| S6 | 7 | How do population size, mutation and fitness design shape evolution? Does GA beat random search? |
| S7 | 8 | Imitation vs evolution: which learns more per unit of experience, and which generalises? |
| S8a/b | 9 | ES vs GA; fixed vs evolving topology. |
| S9 | 10 | RL vs GA vs imitation; how does reward design change behaviour? |
| S10 | 11 | Do hybrids beat their parts? |
| S11 | 12 | Which methods generalise and stay robust, and what does that teach about AI? |

Reports go in `docs/studies/<ID>.md` using the template in `WORKFLOW.md` §4.
