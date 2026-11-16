# Migration audit: the prototype measured against the production app

What it would take to move the redesign into `web/`, measured rather than estimated. Written for
whoever carries out `IMPLEMENTATION_ROADMAP.md`.

- **Audited:** the prototype on branch `ui-2d-redesign`, and the production frontend `web/` at
  `main` commit `1e27c85` (Phase 7 complete, API `v1.25`). The branch changes nothing in `web/`,
  `src/`, `docs/` or `tests/`: `git diff main -- web src docs tests` is empty.
- **How:** both codebases read; the prototype run on both sources; `npm run live-check` (every
  driver, look back, what-if, a GA race, the three Experiment tools, against a server with 334 run
  folders); a static scan of every `data-testid` the browser tests use against both codebases
  (§6). Date: 2026-11-16.

Contents: 1 Summary · 2 What the prototype takes from production · 3 What must not move ·
4 Duplicates · 5 What the live source proved · 6 Test hooks · 7 Production changes the migration
needs · 8 Findings in production code · 9 Open risks

---

## 1. Summary

1. **The data question is settled.** With the live source, every number the redesign shows comes
   from Python through the production API as it is today, for every driver, track and run, with
   two exceptions that were already known and are planned: per-layer contributions ("Flow" and
   routes back to the inputs, `PRODUCTION_MAPPING.md` §6.1) and the exam of every method on one
   chart (roadmap task 8.1). No API change is needed for anything else.
2. **The prototype already runs on production's client layer.** It imports 13 production modules
   read-only (§2): the generated types, the REST and socket client, `useLive`, `useSelection`,
   `usePlayback`, the runs-watch store, and pure helpers. Its live views therefore obey the app's
   linking rules (`ARCHITECTURE.md` §8.2) by construction, not by imitation.
3. **The risk is in the test hooks and in three production refactors, not in the data.** The
   browser tests use 233 distinct test ids; the prototype carries 103 of them (§6). The rest must
   be kept or deliberately changed, view by view. Three production files must be split first
   (§7).
4. **Building against real data found real problems,** all fixed in the prototype: a stale-history
   race when switching runs (also present in production's Timeline, harmlessly, §8.1), overlapping
   ray labels, a play button that disabled itself while playing, chart ticks that only fitted a
   30-generation run, look-back bars that read as a barcode at 30 steps per second.

---

## 2. What the prototype takes from production

All through the `@prod` alias (`vite.config.ts`, `tsconfig.json`), read-only. Type-checking the
prototype type-checks these files too, under the same compiler settings as `web/`.

| Production module | What is used | By |
|---|---|---|
| `api/generated.ts` | Message and REST types (11 importers) | `source/model.ts` and every live component |
| `api/useLiveConnection.ts` | The socket's whole life: connect, reconnect, reload the scenario, fetch tracks and activation curves | `App.tsx` (`LiveLink`) |
| `api/ws.ts` | `connectLive` for `/ws/runs` | `source/runs.ts` |
| `api/rest.ts` | `fetchRuns`, `fetchRunDetail`, `fetchEvolution`, `fetchGenerationStarts`, `fetchAncestry`, `fetchMutation`, `fetchCheckpoints`, `fetchCheckpointDiff`, `fetchExamCard`, `fetchCompare`, `fetchConfigs`, `launchRun`, `stopRun` | `source/live.ts`, `source/runs.ts`, the Experiment tools |
| `state/live.ts` | `useLive`: `loadScenario`, `send`, `tracks`, `recording`, `error`, `activations` | `source/live.ts`, `source/world.ts`, `EnvironmentView`, `Inspector`, `NeuronPanel`, `Timeline` |
| `state/selection.ts` | `useSelection` and its server-following subscription | `source/live.ts` (the bridge and the mirror) |
| `state/playback.ts` | `usePlayback`, `activeMoment` | `source/live.ts`, `source/world.ts` |
| `state/runsWatch.ts` | `applyRunsMessage`, `watchRun`, `runsNewestFirst`, `statusLabel`, `plottableKeys`, `defaultMetric`, `series` | `source/runs.ts`, `Timeline`, `Training` |
| `views/environment/population.ts` | `rankBand`, `rankPosition`, `defaultGhost` | `cars.ts`, `EnvironmentView`, `Timeline` |
| `views/environment/useDrivingKeys.ts` | Keyboard driving | `EnvironmentView` |
| `views/network/networkLayout.ts` | `edgeWidth`, `edgeOpacity` (so the tested `stroke-width` stays) | `encode.ts` |
| `views/network/diff.ts` | `changeStroke`, `mutationStroke` | `encode.ts` |
| `views/network/curve.ts` | `curveBounds`, `plotX`, `plotY`, `curvePoints`, `offChart`, `RANGE_TEXT` | `Inspector` (activation plot), `NeuronPanel` |
| `views/network/inspect.ts` | `rankedReceipt`, `topAndRest`, `sliderRange` | `Inspector` |

Consequence for the migration: these modules are already exercised by the redesigned components
and stay as they are. No step of the plan rewrites them.

---

## 3. What must not move into `web/`

| Prototype file | Why it exists | Why it must stay out |
|---|---|---|
| `src/mock/mockServer.ts` | Stands in for Python with the recorded source: frame interpolation, ray casting for unrecorded cars, a forward pass for what-if, a 2 s rollout, weight × value per connection, activation sampling, quartiles of exam cards | Every one of these is simulation, network or statistics maths; in the app Python computes them (CLAUDE.md §4) |
| `src/mock/types.ts`, `public/recorded/`, `scripts/capture-mock-data.mjs` | The recorded files and their shapes | Production reads the server |
| `src/source/recorded.ts` | The recorded adapter | As above |
| `src/source/live.ts` `startLive` (the bridge and the selection mirror), `src/state/store.ts` | Let one set of components run on two sources | Production components call the production stores directly |
| `src/ui/Notes.tsx` | The design-notes layer | Prototype commentary |
| `scripts/screenshots.mjs`, `scripts/live-check.mjs` | Checks of the prototype | The app has its own suites; the live check's scenarios are a useful checklist for the hand checks of each step (§5) |

What **may** move, and is a decision (D2 in the roadmap): `buildLive` in `src/source/live.ts` is
a pure selector over `useLive`, `usePlayback` and `useSelection`. It does no maths: it picks "the
selected car at the shown moment" from either the frame or the `moment` answer, and drops a
what-if answer that is not for the car and step on screen. Production repeats that choice in
`Hud.tsx`, `draw.ts` (through `EnvironmentView`), `NetworkView.tsx` and `WhatIfPanel.tsx` today.

---

## 4. Duplicates: the same job in both codebases

Each pair is resolved during the migration step that touches it (roadmap column).

| Prototype | Production | Difference | Resolution | Step |
|---|---|---|---|---|
| `views/environment/camera.ts` (`fitCamera`, `worldToScreen`, `zoomAt`, `panBy`, `wheelFactor`, plus `easeCamera`, `lookAhead`, margins) | `views/environment/geometry.ts` | The prototype adds easing, look-ahead and per-side margins | Add the three to `geometry.ts` with tests; keep its functions | E3 |
| `views/environment/scene.ts`, `trackArt.ts` | `views/environment/draw.ts` | A new drawing approach (world transform, layers, textures) on the same inputs | Replace `draw.ts` layer by layer | E1, E2 |
| `views/environment/cars.ts` (`bodyOf`, liveries, `drawOrder`, `nameTag`) | `population.ts` (`carRole`, `roleLabel`, bands) | Liveries are new; the band maths is production's already | Move liveries next to `population.ts` | E2 |
| `views/network/encode.ts` (`layoutNodes`, `edgePath`, `strongPaths`, `receiptPaths`) | `networkLayout.ts`, `diff.ts` | Curves and a new layout; strokes are now production's own | Extend `networkLayout.ts`; layout tests change with it | N1 |
| `views/network/Inspector.tsx` `Why` | `OutputInspector.tsx` | Same receipt helpers; a different layout and the activation plot | Restyle `OutputInspector` | N2 |
| `views/network/NetworkCompact.tsx` | `NetworkCompact.tsx` | Same content, restyled | Restyle in place | N3 |
| `views/network/NeuronPanel.tsx` | `NeuronView.tsx` | The prototype's is shorter (no input diagram, no dial) | Restyle `NeuronView`; keep its 35 tested hooks | N3 |
| `views/experiment/charts.tsx` (`linear`, `niceTicks`, `linePath`, `bandPath`, `Plot`, axes, `Figure`) | `charts.ts` helpers; a `viewBox` per chart | The prototype draws at real size | One shared `Plot`; `charts.ts` stays | X1 |
| `views/experiment/RunTimeline.tsx` charts and `GaDashboard` | `Evolution.tsx`, `Ancestry.tsx`, `Timeline.tsx` | One page instead of one column; generation panels follow the cursor | Split `Evolution.tsx`, re-lay out | X2 |
| `source/runs.ts` `useRunsSocket` | The same socket code inside `Timeline.tsx` and `TrainingRuns.tsx` | The prototype has one hook with a stale-history guard (§8.1) | One hook in `web/src/state/` | X2 |
| `ui/controls.tsx` | Each view's own `<button>`, `<select>`, bars | Shared components | `web/src/components/ui/` | S2 |
| `state/store.ts` (layout part) | `state/layout.ts` | Adds `theme`, `drawerTab` | Two fields, sanitised and tested | S1 |

---

## 5. What the live source proved

Everything below ran against the real server with `npm run live-check` and was looked at in the
screenshots in `screenshots/live/`.

| Element | Comes from | Checked |
|---|---|---|
| Track, kerbs, sand, boards, start line | `GET tracks/{name}`; decoration computed by `trackArt` | All six tracks loaded with the expert and looked at in Whole track view (a separate probe, not in `live-check`) |
| Cars, statuses, liveries, rank colours, elites, ghost | `frame.cars`, `scenario.population` | 1, 8, 100 cars; a 20-network generation with its ghost |
| Rays and their labels | `frame.selected.rays` | Every driver |
| Telemetry, step, look back | `frame`, `trail`, `moment` (`inspect_step`) | Look-back to a third of the remembered window, Back to now |
| Keyboard driving and recording | `key_state`, `record_start`, `record_stop`, `recording` | Speed above 1 m/s after a second of the up arrow; Record bar present |
| Network diagram, inspector, sensitivity | `network`, `selected.trace`, `selected.explain` | Car 43 of 100 random networks |
| What-if and its ghost | `what_if` → `explanation` (with `ghost` from Python's physics) | 61-point ghost path |
| Neuron driver | `selected.neuron`, `set_param` | `w_left` 0 → 5 round trip |
| Lineage, mutation | `GET .../rows/{r}/mutation` | Banner and Mutation mode on a child |
| Checkpoint change | `GET checkpoints`, `GET .../diff` | "Change" mode in the replay of a random-search checkpoint (a separate probe) |
| Timeline, GA | `GET evolution`, `generations/{g}/starts`, `.../ancestry`, `exams/{u}`, `/ws/runs` | A 117-generation run at generation 5; an outcome cell raced its start |
| Timeline, other methods | `/ws/runs` metrics, checkpoints, exams | A random-search run |
| Compare | `GET runs`, `GET runs/compare` | The newest sweep, medians and bands |
| Training | `GET configs`, `/ws/runs`; `POST runs`, `.../stop` exist and are wired | List and chart (no run launched by the check) |

Not proved at all yet: launching and stopping a run from the redesigned Training tool (the code
calls production's `launchRun` and `stopRun`, but no run was started during this audit), and
"Replay champion" from the GA dashboard. Both are on the hand-check list of X3.

**Performance**, observed, not measured: 100 cars at 30 frames a second with the new world drawing
showed no visible stutter on the development machine. Frame time was not profiled; the roadmap's
E1 measures it before the step can pass.

---

## 6. Test hooks

The 18 browser specs in `web/e2e/` find elements by `data-testid` and read `data-*` attributes.
A static scan (template ids as patterns) found **233 distinct ids** used by the specs:

| | Ids the specs use that the code renders |
|---|---|
| Production `web/src` | 227 of 233 (the other 6 are built in ways a static scan does not see) |
| Prototype | **103 of 233** (44 %) |

Per spec, the prototype's coverage and the ids it lacks:

| Spec | View | Prototype has | Missing (first six) |
|---|---|---|---|
| `inspector.spec.ts` | Network | 22 of 22 | none |
| `ghost.spec.ts` | Environment + Network | 15 of 16 | `car-label` |
| `linking.spec.ts` | all three | 21 of 25 | `car-label`, `cnet-cell-1-5`, `network-compact-readout`, `network-view-compact` |
| `trail.spec.ts` | Environment | 20 of 26 | `hud-past`, `select-speed`, `trail-legend`, `trail-live`, `trail-readout`, `trail-slider` |
| `record.spec.ts` | Environment | 12 of 14 | `dataset-select`, `dataset-summary` |
| `network.spec.ts` | Network | 17 of 23 | `car-label`, `cnet-input-0`, `cnet-input-6`, `cnet-open-full`, `network-view-compact`, `neuron-view` |
| `expert.spec.ts` | Environment | 6 of 8 | `expert-note`, `select-speed` |
| `environment-camera.spec.ts` | Environment | 5 of 8 | `camera-free`, `compact-follow`, `connection-badge` |
| `population.spec.ts` | Environment + Experiment | 14 of 19 | `hud-race`, `hud-race-role`, `select-speed`, `timeline-label`, `timeline-run` |
| `smoke.spec.ts` | shell | 19 of 30 | `companion-drawer`, `connection-badge`, `page-environment`, `page-experiment`, `page-network`, `panel-environment` and 5 more |
| `timeline.spec.ts` | Experiment | 13 of 25 | `live-cursor`, `network-change-legend`, `select-speed`, `show-change`, `timeline-card-row`, `timeline-card-shifted` and 6 more |
| `training.spec.ts` | Experiment | 8 of 15 | `exam-row`, `live-exam-point`, `live-hover`, `live-point`, `training-run-status`, `training-run-units` and 1 more |
| `evolution.spec.ts` | Experiment | 14 of 29 | `evolution`, `evolution-distribution-hit`, `evolution-distribution-hover`, `evolution-fitness-hit`, `evolution-fitness-hover`, `evolution-fitness-legend` and 9 more |
| `compare.spec.ts` | Experiment | 6 of 13 | `compare-crosshair`, `compare-final-median`, `compare-hover`, `compare-loading`, `compare-run-line`, `compare-skipped` and 1 more |
| `lineage.spec.ts` | Experiment + Network | 5 of 16 | `ancestry`, `ancestry-hit`, `ancestry-hover`, `ancestry-race-1`, `ancestry-race-2`, `ancestry-row` and 5 more |
| `neuron.spec.ts` | Network | 13 of 35 | `curve-dot`, `neuron-empty`, `neuron-input-0`, `neuron-node`, `neuron-output`, `neuron-range` and 16 more |
| `datasets.spec.ts` | Experiment | 1 of 17 | Datasets is not built in the prototype |
| `toylab.spec.ts` | Experiment | 0 of 17 | Toy lab is not built in the prototype |

During this task the prototype took over production's ids wherever its element does the same job
(`environment-canvas`, `select-track`, `select-policy`, `select-car`, `car-pick`, `hud-status`,
`camera-follow`, `camera-fit`, `experiment-tab-*`, `launch-config`, `hud-speed`, `hud-step`, …),
which raised coverage from 89 to 103.

**Deliberate departures**: the redesign changes the element, so the test must change with the
code, in the same commit, and the step's report must say so:

| Test hook | Today | Redesign | Specs that use it |
|---|---|---|---|
| `select-speed` | A `<select>` | A segmented control | `expert`, `population`, `smoke`, `timeline`, `trail` |
| `trail-slider`, `trail-readout`, `trail-live` | A range input and a read-out | The look-back strip (drag on the steer and throttle history) | `trail` |
| `hud-race`, `hud-race-role`, `hud-past` | Rows of the HUD table | Groups of the telemetry strip | `population`, `trail` |
| `network-view-compact`, `cnet-*` | The compact network in drawers | The same, restyled: ids can stay | `linking`, `network` |
| `show-change`, `show-mutation` | Checkboxes in banners | Options of "Lines show" | `timeline`, `lineage` |
| Edge `stroke-width` | `edgeWidth` | `edgeWidth` (the prototype now uses production's): no change | `network` |

---

## 7. Production changes the migration needs

Found by building the prototype against production. None changes behaviour; each makes a later
step smaller.

1. **Extract the network hooks.** `useMutation` and `useCheckpointChange` are private functions in
   `web/src/views/network/NetworkView.tsx`. The redesigned rail, the breadcrumb and the Timeline
   need the same answers. Move them to `web/src/views/network/hooks.ts` and export them; the
   prototype had to write its own (`refreshOrigin` in `src/source/live.ts`).
2. **One `/ws/runs` hook.** `Timeline.tsx` and `TrainingRuns.tsx` each open and manage the socket
   with the same code. One `useRunsSocket(watched)` in `web/src/state/` (the prototype's, with the
   guard of §8.1) removes the copy.
3. **Split the three big files before restyling them**: `styles.css` (2,271 lines) into tokens,
   base and one file per view; `draw.ts` into scene, world, cars, overlays; `Controls.tsx` (427
   lines) into scenario, transport, camera and record bars. Each split keeps behaviour and every
   test hook (`PRODUCTION_MAPPING.md` §8).
4. **Test ids on shared controls.** The prototype's `Select` takes `testId` and `Segmented` takes a
   `testId` per option; the shared components in `web/src/components/ui/` need the same, or the
   specs' `select-*` and `camera-*` hooks are lost when views move onto them.
5. **Optional: a world selector** (`buildLive` as `web/src/state/world.ts`, §3). A decision, D2.

---

## 8. Findings in production code

Production is not changed on this branch. These are reported for the steps that touch the files.

### 8.1 Timeline: one request for the previous run's checkpoint when switching runs

`web/src/views/experiment/Timeline.tsx` resets the watched run's history in an effect
(`setWatch((s) => watchRun(s, runId))`), but on the render in which `runId` has already changed
the history still holds the previous run's checkpoints and exams. The exam-card effect, which
depends on `[runId, unit, hasCard]`, then runs with the new run id and the old run's unit, and
fetches `GET runs/{new run}/exams/{old unit}`. The answer is dropped (the next render clears the
unit and the effect's cleanup sets `live = false`), so nothing wrong is shown; the server logs a
404. Found by code reading after the prototype, which copied the pattern, showed exactly this 404
in `live-check` (`/api/runs/…-random_search-e2estop/exams/117` right after a GA run). The
prototype's `useRunsSocket` never returns history whose `watched` differs from the run asked for;
item 2 of §7 carries that guard into production.

### 8.2 Smaller observations

- `/ws/runs` reports runs that are running or were launched since the server started, not every
  run folder. Production's Training tool shows an empty list with no explanation in that case; the
  redesign says where finished runs are.
- `GET /api/runs` takes seconds with about 330 run folders (it reads each folder's metadata). The
  prototype caches the list for 30 s across tools; production fetches it in every tool on every
  visit. A server-side cache or pagination would be a later, additive API change.
- GA runs do not always store an exam card at every checkpoint (the S6 runs examined only their
  last unit). The redesigned exam panel shows the newest card at or before the chosen generation
  and offers to jump to the first one; production's Timeline already handles "no card" per unit.
- With 100 cars the production HUD and legend cover much of the Whole track view
  (`UI_AUDIT.md` E4); with the redesign's margins the road stays clear of the floating panels on
  `train_loop_01`. The other five tracks are on E3's hand-check list.

---

## 9. Open risks

| Risk | Where | How the plan keeps it small |
|---|---|---|
| Frame time of the new world drawing with 100 cars was not profiled | `scene.ts` → `draw.ts` | E1 measures 1, 20 and 100 cars before and after; offscreen-canvas cache of the static world if needed |
| Track decoration looked at on all six tracks at Whole track scale only, not close up | `trackArt.ts` | E1 unit tests on all six tracks: no kerb or sand point inside the road; close-up screenshots of every corner type |
| A camera that eases changes when "settled" for tests that click cars by position | `environment-camera`, `linking`, `ghost`, `trail` specs | E3: `data-camera-settled`; no easing under `prefers-reduced-motion` (the prototype already does this), and `reducedMotion: "reduce"` added to `web/playwright.config.ts`, which does not set it today |
| Two components written for two sources hide a source-specific bug | every view | The migration ports the live branch only; the recorded branch is not ported (§3) |
| Fonts change text widths | everywhere | S1 lands fonts with tokens and reviews every view at 1280 and 390 px |
