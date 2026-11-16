# Migration plan: from the prototype into the app

How the redesign moves into `web/`, one view at a time, so each view can be tested and released on
its own. **Nothing here has been started.** It is separate from `docs/ROADMAP.md`: it adds no
learning algorithm, changes no contract of `ARCHITECTURE.md` §6 and does not reorder the phases.

It rests on two documents: `MIGRATION_AUDIT.md` (what was measured: reuse, duplicates, test hooks,
findings) and `PRODUCTION_MAPPING.md` (where each recommendation G1…X18 goes, file by file).

---

## 0. Decisions before the first step

These are the user's to make; each needs an ADR in `docs/DECISIONS.md` once made
(CLAUDE.md §3.6).

| # | Question | Options | Recommendation |
|---|---|---|---|
| D1 | How do old and new coexist during the migration? | **(a) In place:** each step replaces the old look in the same files, and the app's tests must pass after every step. **(b) Side by side:** new views in new folders behind a switch (`?ui=2`) until all are done, then the old ones are deleted. | **(a).** It fits the project's one-task-one-commit rule, leaves no dead code, and the 80 browser tests guard every step. Undoing a step is reverting one commit. (b) makes comparison easy but doubles views and tests for weeks, and the suite can only test one of the two. |
| D2 | Should production get the prototype's world selector (`buildLive` → `web/src/state/world.ts`)? | **Yes:** one pure selector answers "the selected car at the shown moment" and "the what-if answer for this car and step". **No:** components keep reading the three stores directly. | **Yes, in step S2.** Production makes that choice in four places today (`Hud.tsx`, `EnvironmentView.tsx`, `NetworkView.tsx`, `WhatIfPanel.tsx`), and the prototype's components already read it. It adds a state module, so `ARCHITECTURE.md` §8.1 is updated in the same step. |
| D3 | Bundle the IBM Plex fonts (three npm packages, SIL OFL)? | Yes / keep the system font | Yes: the identity rests on them, and nothing is fetched at runtime. |
| D4 | Add a theme setting (light, dark, follow the system) to the layout store? | Yes / keep "follow the system" only | Yes: `UI_AUDIT.md` A4. |
| D5 | Branches | One branch per step / one per view group | **One per view group**: `visual-shell`, `visual-environment`, `visual-network`, `visual-experiment`, `visual-small`; one commit per step; merged into `main` at the group's gate. A view group is the unit that can be released. |

---

## 1. Order

```
 S0 Groundwork ──► S1 Identity ──► S2 World selector ─┬─► E1 ► E2 ► E3        Environment ▸ gate
 (no visual change)  (shell)        (no visual change) ├─► N1 ► N2 ► N3        Network     ▸ gate
                                                       └─► X1 ► X2 ► X3 ► X4   Experiment  ▸ gate
                                                                       R1 Small windows and All layout
 Future, not visual stages:  F1 Flow and routes to the inputs (API addition)  ·  F2 exam strip (task 8.1)
```

| Step | Name | Visual gain | Difficulty | Category (mapping §1) |
|---|---|---|---|---|
| S0 | Groundwork: stylesheet split, shared controls with test ids, extracted hooks, one `/ws/runs` hook | none, on purpose | medium | 2 |
| S1 | Identity: palette, type, top bar, breadcrumb, theme switch, drawer | high, everywhere | medium | 2 |
| S2 | World selector (D2) | none, on purpose | low | 2 |
| E1 | The track | the highest | medium to high | 1 + 2 |
| E2 | Cars, rays, trail, overlays | high | medium | 1 |
| E3 | The stage: telemetry, bars, minimap, standings, look-back strip, camera | high | high | 1 + 2 |
| N1 | The diagram | high | medium | 1 |
| N2 | The inspector rail | medium, large in use | medium | 2 |
| N3 | Compact network and the neuron driver | medium | low | 1 |
| X1 | Chart kit | medium, on every chart | medium to high | 2 |
| X2 | Timeline dashboard | high | medium | 2 |
| X3 | Compare and Training | medium | medium | 1 + 2 |
| X4 | Toy lab and Datasets | medium | medium | 1 + 2 |
| R1 | Small windows, the All layout | medium | low to medium | 2 |

Why this order:

- **S0 to S2 first** because every view stands on them, and because they change nothing visible:
  their test is that every screenshot stays the same. A later visual step then changes one thing.
- **Environment first** among the views: it is the weakest screen today (`UI_AUDIT.md` §2.2) and
  sets the app's character. Its steps share `draw.ts` and `EnvironmentView.tsx`, so they run one
  after another.
- **Network and Experiment do not depend on the Environment** and can run in either order after
  S2, but not at the same time as a step that touches the same files.
- **The smallest worthwhile release** is S0 + S1 + E1 + E2: the app has its identity and the track
  looks like the prototype's, with every panel where it is today.

---

## 2. How every step is checked

**The standing checks**, after every step:

1. `cd web && npm run build` (type-check and build) passes.
2. `cd web && npm run test` passes with no test removed.
3. The browser specs of the step's view, listed per step, pass with no test removed or weakened.
   A test changes only where the step's table says its hook changes, in the same commit as the
   code (`MIGRATION_AUDIT.md` §6, "deliberate departures").
4. Screenshots of every screen the step touched, light and dark, 1440 × 900 and 390 × 844, with no
   console errors, compared side by side with the prototype's (`screenshots/` and
   `screenshots/live/`). Every difference is either listed in the step's report as intended, or a
   bug.
5. The step's hand checks, done with `nd serve` running: the scenarios of
   `scripts/live-check.mjs` that touch the step's view, by hand, in the app.

**The view gate**, after a view's last step, before its branch is merged into `main`:

1. The full browser suite: `cd web && npm run e2e`.
2. `uv run pytest` (Python is not touched by any visual step; it must pass unchanged).
3. Every function of the view listed in `UI_AUDIT.md` §4 exercised once by hand, ticked off.
4. The user looks at the view and accepts it.

**Undo:** each step is one commit on its group's branch; `git revert` of that commit restores the
previous step. A group not yet merged can be abandoned without touching `main`.

---

## 3. The steps

In each step, "Lift" means the prototype file is the starting point and is adapted to production's
stores; "Keep" means production code that must survive unchanged in behaviour. Nothing under
`src/mock/` or `src/source/recorded.ts` is lifted (`MIGRATION_AUDIT.md` §3).

### S0 · Groundwork (no visual change)

| | |
|---|---|
| **Objective** | Make the frontend ready to change, without changing how anything looks or behaves. |
| **Work** | (a) Split `web/src/styles.css` into `web/src/styles/tokens.css`, `base.css` and one file per view, introducing today's values as tokens (`PRODUCTION_MAPPING.md` §8.1). (b) Shared controls in `web/src/components/ui/` with test-id props: `Button`, `Segmented` (a test id per option), `Select` (`testId`), `Field`, `Chip`, `Panel`, `Stat`, `ValueBar`, `Icon`; not yet used. (c) Move `useMutation` and `useCheckpointChange` out of `NetworkView.tsx` into `web/src/views/network/hooks.ts` (audit §7.1). (d) One `useRunsSocket(watched)` in `web/src/state/` with the stale-history guard, used by `Timeline.tsx` and `TrainingRuns.tsx` (audit §7.2, §8.1). |
| **Lift** | `src/ui/controls.tsx`, `src/ui/Icon.tsx` (components only; styles stay today's); `useRunsSocket` from `src/source/runs.ts`. |
| **Keep** | Every component's output and every test hook. |
| **Tests** | All Vitest; the full browser suite (the stylesheet touches every view). New unit test: `useRunsSocket` never returns another run's history. |
| **Hand checks** | Switch runs in the Timeline: no 404 in the server log. |
| **Done when** | Screenshots of every view, tool and layout before and after are identical (anti-aliasing aside). |

### S1 · Identity

| | |
|---|---|
| **Objective** | Colours, type and frame. |
| **Recommendations** | G1 to G8, G10 to G12. |
| **Files** | `web/src/styles/tokens.css`, `base.css`; `web/package.json` (fonts, D3); `web/src/main.tsx`; `TopBar.tsx`, `ConnectionBadge.tsx` (keep `connection-badge`, `data-status`), `CompanionDrawer.tsx` (keep `companion-drawer`); new `Breadcrumb.tsx`; `web/src/nav/views.ts`; `web/src/state/layout.ts` + `layout.test.ts` (`theme`, `drawerTab`, D4); `App.tsx`. |
| **Lift** | `src/theme/tokens.css`, `base.css`; `TopBar`, `Crumbs`, `Drawer` from `src/App.tsx` (without the source switch). |
| **Specs** | `smoke`, then the full suite (the shell touches every page). |
| **Hand checks** | Theme switch in all three positions, remembered after reload; breadcrumb for each driver (keyboard, random networks, checkpoint replay, GA race). |
| **Note** | The canvas still reads page tokens until E1: give the `--w-*` tokens today's canvas colours here, so the track does not change in this step. |

### S2 · World selector (D2; no visual change)

| | |
|---|---|
| **Objective** | One answer to "what is on screen for the selected car right now". |
| **Files** | New `web/src/state/world.ts` (+ `world.test.ts`); `docs/ARCHITECTURE.md` §8.1; ADR. Components are not changed yet: later steps switch to it as they are rebuilt. |
| **Lift** | `buildLive` and the `World` shape from `src/source/live.ts` and `src/source/model.ts`. Not `startLive` (the bridge exists only for the prototype). |
| **Tests** | Unit tests: the moment wins over the frame only for the same car and step (the cases of `playback.test.ts`); a what-if answer for another car or step is dropped (the cases of `live.test.ts`). |

### E1 · The track

| | |
|---|---|
| **Recommendations** | E1 to E4, E10. |
| **Files** | `web/src/views/environment/draw.ts` → `scene.ts`, `drawWorld.ts` (cars and overlays still drawn the old way, moved to `drawCars.ts`, `drawOverlays.ts`); new `trackArt.ts` + `trackArt.test.ts`; `tokens.css` (`--w-*`). |
| **Lift** | `src/views/environment/scene.ts` (the world layers and textures), `trackArt.ts`. |
| **Keep** | The `Scene` inputs; every `data-*` on `environment-canvas`. |
| **Specs** | `environment-camera`, `linking`, `trail`, `ghost`, `population`. |
| **Measure** | `drawScene` frame time with 1, 20 and 100 cars before and after; the live view must stay within 33 ms per frame (`ARCHITECTURE.md` §9). If not: draw the static world to an offscreen canvas, redrawn only when the camera moves. |
| **Hand checks** | All six tracks, Whole track and close up, day and night (the probe in `MIGRATION_AUDIT.md` §5 did Whole track only). Rays still end exactly on the white lines. |

### E2 · Cars, rays, trail, overlays

| | |
|---|---|
| **Recommendations** | E5 to E9, E17; liveries (DESIGN_SPEC §7.2). |
| **Files** | `drawCars.ts`, `drawOverlays.ts`, `scene.ts`; new `cars.ts` next to `population.ts` (liveries, draw order, name tags); `EnvironmentView.tsx` (legend markup only); `tokens.css` (`--car-*`, `--ray-*`, `--track-*`). |
| **Lift** | The car, ray, label-placement, trail, what-if and look-back layers of `scene.ts`; `src/views/environment/cars.ts`. |
| **Specs** | `population`, `ghost`, `trail`, `linking`, `expert`, `record`. |
| **Hand checks** | Every driver: keyboard, constant, neuron, random networks (1, 8, 20, 100 cars), expert, checkpoint replay, GA generation with ghost; every car state: running, crashed, finished, stalled, out of time, selected, elite, ghost, look-back, what-if ghost. The rank ramp and liveries re-run through the dataviz validator on both asphalts. |

### E3 · The stage

| | |
|---|---|
| **Recommendations** | E11 to E16, E18, G8 (Standings tab). |
| **Files** | `Hud.tsx` → `Telemetry.tsx`; `Controls.tsx` → `ScenarioBar.tsx`, `Transport.tsx`, `CameraBar.tsx`, `RecordBar.tsx` (audit §7.3); `Trail.tsx` → the look-back strip; new `Minimap.tsx`, `Standings.tsx`; `geometry.ts` (+ tests: `easeCamera`, `lookAhead`, margins per side); `EnvironmentView.tsx`; `CompanionDrawer.tsx`; `environment.css`; `web/playwright.config.ts` (`reducedMotion: "reduce"`). |
| **Lift** | `Telemetry.tsx`, `Extras.tsx` (minimap, legend, look-back strip, standings), the bars in `EnvironmentView.tsx` (including `CarSelect` and `RecordBar`), `camera.ts`. |
| **Keep** | `IntField`, `ghostAfterMove`, `useDrivingKeys`, `status.ts`, `trailLayout.ts`, the click and drag rules of task 4.10; every `hud-*` id with the same text formats (`hud-step` is read 48 times). |
| **Tests that change** | `select-speed` (segmented); `trail-slider`, `trail-readout`, `trail-live` (the strip); `hud-race`, `hud-race-role`, `hud-past` (telemetry groups); camera expectations where look-ahead moves the centre. Each changed in the same commit as its code. |
| **Specs** | `environment-camera`, `expert`, `ghost`, `linking`, `population`, `record`, `trail`, `smoke`; the click-a-car specs run 20 times in a row without a failure. |
| **Hand checks** | Drive a lap with the keyboard; record, save and discard a demonstration; race a GA generation and change generation, start and ghost; look back by the strip and by clicking the trail; what-if with the ghost; Whole track on all six tracks with no floating panel over the road. |
| **Environment gate** | §2. |

### N1 · The diagram

| | |
|---|---|
| **Recommendations** | N1 to N6, N11, N12, N13. |
| **Files** | `NetworkDiagram.tsx`, `networkLayout.ts` (+ `networkLayout.test.ts`: 13 expectations change with `layoutNodes`), `diff.ts` (`originText` gains ranks and counts), `NetworkView.tsx` ("Lines show" replaces the two module-level booleans; `show-change` and `show-mutation` become the options' ids), `network.css`. |
| **Lift** | `src/views/network/Diagram.tsx`, `encode.ts` (`layoutNodes`, `edgePath`, `receiptPaths`); the strokes are already production's (`edgeWidth`, `changeStroke`, `mutationStroke`). Not `strongPaths` over all layers: it needs F1. |
| **Specs** | `network`, `inspector`, `lineage`, `ghost`, `timeline`, `linking`. |
| **Hand checks** | Layer sizes [7, 2], [7, 8, 2], [7, 16, 16, 2], [7, 64, 64, 2] and four hidden layers; nothing selected, a hidden neuron, an output, a connection; Change in a checkpoint replay; Mutation for a child; tooltip numbers equal `data-weight` / `data-activation`. |

### N2 · The inspector rail

| | |
|---|---|
| **Recommendations** | N7 to N10, N15. |
| **Files** | `NetworkView.tsx` (two columns), `OutputInspector.tsx` (output cards, receipt, activation plot from `GET activations`), `WhatIfPanel.tsx` (locked states with their action), new `ActivationPlot.tsx`, `OutputCards.tsx`; `selection.ts` (+ tests) for the transient `peek`; `network.css`. |
| **Lift** | `src/views/network/Inspector.tsx` (it already uses `rankedReceipt`, `topAndRest`, `sliderRange` and the curve helpers). |
| **Specs** | `inspector` (the prototype already carries all 22 of its hooks), `ghost`, `neuron`, `timeline`, `lineage`. |
| **Hand checks** | At 1280 × 720 the output cards, the receipt and the sum are visible without scrolling; the activation plot's point equals the receipt's z and value for ten random frames. |

### N3 · Compact network and the neuron driver

| | |
|---|---|
| **Recommendations** | N16, N17. |
| **Files** | `NetworkCompact.tsx` (restyle; ids stay), `NeuronView.tsx`, `ParamSlider.tsx`, `curve.ts` (unchanged). |
| **Lift** | `src/views/network/NeuronPanel.tsx` for the layout; production's `NeuronView` keeps its input diagram, dial and its 35 tested hooks. |
| **Specs** | `neuron`, `network`, `linking`. |
| **Network gate** | §2. |

### X1 · Chart kit

| | |
|---|---|
| **Recommendations** | X1 to X3. |
| **Files** | New `web/src/views/experiment/Plot.tsx`; `ChartHover.tsx`; then one chart per commit: `LossChart`, `LiveChart`, `CompareChart`, `PredictionPlot`, the charts in `Evolution.tsx`, `Ancestry.tsx`, `DatasetView.tsx`, `ToyNetwork.tsx`. `charts.ts` and its tests stay. |
| **Lift** | The components of `src/views/experiment/charts.tsx` (`Plot`, `YAxis`, `XAxis`, `Figure`, `Objective`, `Tip`), and `genTicks` (ticks from the chart's width). |
| **Specs** | Per converted chart: the spec that hovers it (`training`, `compare`, `evolution`, `toylab`, `lineage`, `datasets`). |
| **Hand checks** | Each chart at 400, 800 and 1600 px: tick text the same size; hovering the left edge, middle and right edge reads the first, middle and last unit. |

### X2 · Timeline dashboard

| | |
|---|---|
| **Recommendations** | X4 to X11. |
| **Files** | `Timeline.tsx`; `Evolution.tsx` split into `FitnessChart`, `DistributionChart`, `Histogram`, `DiversityChart`, `StartsMap`, `OutcomeGrid`; `Ancestry.tsx`; new `ExamCard.tsx`; `evolutionCharts.ts`; `experiment.css`. |
| **Lift** | `GaDashboard`, `ExamPanel`, `Slot` and the charts of `src/views/experiment/RunTimeline.tsx`; `LiveGaTimeline` and `LiveRunTimeline` of `Timeline.tsx` (they already use production's REST helpers, `defaultGhost` and the runs-watch helpers). |
| **Keep** | Start shift and "Watch the expert"; the run and track pickers; the per-car rows of the exam card and clicking one to select the car; the "train again" messages of runs trained before tasks 7.6 to 7.8. |
| **Specs** | `timeline`, `evolution`, `lineage`, `population`. |
| **Hand checks** | A GA run, a GA run trained before 7.7, a random-search run, an imitation run, a DAgger run, a baseline, a run still training; the generation panels follow the cursor; an exam panel where the first exam is late (S6 runs); colour-blind check of the outcome fills. |

### X3 · Compare and Training

| | |
|---|---|
| **Recommendations** | X12 (method and sweep; study grouping needs one API field, mapping §6.3), X13, X16. |
| **Files** | `CompareRuns.tsx`, `RunList.tsx`, `CompareChart.tsx`, `TrainingRuns.tsx`, `LiveChart.tsx`, `ExperimentView.tsx`. |
| **Lift** | `LiveCompare` (run list grouped by method and sweep, search, one task per chart, the cached run list) and `LiveTraining` of the prototype. |
| **Specs** | `compare`, `training`. |
| **Hand checks** | Launch, watch and stop a run from the Training tool (not done during the audit); a diverged run's chart shows no clipped label; about 330 runs stay fast to find. |

### X4 · Toy lab and Datasets

| | |
|---|---|
| **Recommendations** | X17, N18 (optional). |
| **Work** | First prototype them (they are specified in `DESIGN_SPEC.md` §8.7 but not built), on the live source like the other tools; then migrate. |
| **Specs** | `toylab`, `datasets`, `record`. |
| **Experiment gate** | §2. |

### R1 · Small windows and the All layout

| | |
|---|---|
| **Recommendations** | G9, N16, `DESIGN_SPEC.md` §12. |
| **Files** | `web/src/styles/*.css` (container queries), `NetworkView.tsx`, `EnvironmentView.tsx`, `App.tsx` (the All layout's minimum panel sizes). |
| **Hand checks** | Every view at 390, 560, 768, 940, 1120, 1440 and 1920 px in Page, Split and All: no horizontal page scroll, no control cut off, the track always visible on the Environment page. |

### F1 · Flow and routes to the inputs (future, an API addition)

Every layer's weight × value for the selected car, sent by Python (`PRODUCTION_MAPPING.md` §6.1):
`Explanation` gains the contributions, API `v1.25` → `v1.26`, ADR, `npm run gen:types`, server
tests (each row sums to z − b to 1e-12, the check of task 2.9). Then "Flow" and `strongPaths` over
all layers come out of the prototype's recorded-only corner.

### F2 · Every method on the Driving Exam (roadmap task 8.1)

The exam strip (`DESIGN_SPEC.md` §8.5) is task 8.1's visual brief. It needs `POST compare`
(`ARCHITECTURE.md` §7, planned): medians and quartiles over seeds are Python's. It is not a visual
stage and is not built before Phase 8 sets the comparison's rules.

---

## 4. How this fits the project's rules

- **One task at a time.** Each step is a task with a Brief and a Report in `task-notes/`, numbered
  apart from the phases (`S0`, `E3`, …) so `docs/CHECKLIST.md` can list them without disturbing
  phase numbers. Adding those rows is a change to the project's documents: the user decides
  whether and when.
- **Contracts.** No step changes `ARCHITECTURE.md` §6 or the API. S2 adds a state module (§8.1,
  with an ADR). F1 changes the API by an addition.
- **Python stays the source of truth.** No lifted component computes a network output, a rollout,
  a statistic or an explanation; the prototype's live branch, which is what is lifted, computes
  none (`MIGRATION_AUDIT.md` §1).
- **Tests are the safety net.** No test is removed; a test changes only with the hook it reads, in
  the same commit, and the step's report lists it.

---

## 5. When the whole migration is done

1. The full standing checks, the full browser suite and `uv run pytest`, all unchanged in count.
2. Every function in `UI_AUDIT.md` §4 exercised once by hand.
3. The same scenario and seed recorded before and after: identical frames (the redesign must not
   have changed what is simulated or sent; compare as `tests/env/test_episode.py` does).
4. The prototype's screenshots and the app's, side by side; every difference intended.
5. `docs/ARCHITECTURE.md` §8 describes the shell, the breadcrumb, the theme setting, the drawer
   tabs and the world selector as built.
6. The prototype is kept as the design reference, or removed with an ADR; its recorded source no
   longer has a job once the app has every screen.
