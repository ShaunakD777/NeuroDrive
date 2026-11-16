# Visual implementation roadmap

A staged plan for bringing the design in this folder into the real NeuroDrive app, **if and when
it is approved**. Nothing here has been started.

This plan is separate from `docs/ROADMAP.md`. It adds no learning algorithm, changes no contract
and does not reorder the project's phases. It can be taken up between phases, or one stage at a
time beside them; §4 says how it fits the project's working rules.

Recommendation numbers (G1, E5, N7, X4 …) refer to `PRODUCTION_MAPPING.md` §2.

---

## 1. Order, and why

Stages are ordered by three things: what later stages stand on, how much the eye gains, and how
much can go wrong.

| Stage | Name | Stands on | Visual gain | Difficulty | Category of work |
|---|---|---|---|---|---|
| V0 | Groundwork: tokens and stylesheet split | — | none (on purpose) | medium | 2 |
| V1 | Identity: palette, type, shell | V0 | high, everywhere | medium | 2 |
| V2 | The track | V0 (V1 for final colours) | **highest** | medium to high | 1 + 2 |
| V3 | Cars and what is drawn on the track | V2 | high | medium | 1 |
| V4 | The stage: telemetry, controls, minimap, look-back, camera | V1, V3 | high | high | 1 + 2 |
| V5 | Network diagram | V1 | high | medium | 1 |
| V6 | Network inspector rail | V5 | medium | medium | 2 |
| V7 | Chart kit | V1 | medium, on every chart | medium to high | 2 |
| V8 | Timeline dashboard | V7 | high | medium | 2 |
| V9 | Compare, Training, Toy lab, Datasets | V7 | medium | medium | 1 + 2 |
| V10 | Small windows and the All layout | V4, V6, V8 | medium | low to medium | 2 |
| F1 | Flow mode and routes to the inputs | V5, a server change | medium | medium | 3 |
| F2 | Every method on the Driving Exam | V7, roadmap task 8.1 | high | — (belongs to Phase 8) | 3 |

The Environment view is first after the groundwork (V2 to V4) because it is the weakest screen
today and the one that defines the app's character. V5 and V7 do not depend on it and can be done
in parallel or swapped in if a break from canvas work is wanted. Do not start two stages that
touch the same files at once: V2, V3 and V4 share `draw.ts` and `EnvironmentView.tsx`.

**The smallest worthwhile slice** is V0 + V1 + V2 + V3: the app has its identity and the track
looks like the prototype's, with every panel still where it is today.

---

## 2. The stages

Every stage ends with the same four checks, called **the standing checks** below:

1. `cd web && npm run build` (type-check and build) passes.
2. `cd web && npm run test` passes with no test removed.
3. `cd web && npm run e2e` passes with no test removed or weakened. A test may change only where
   this plan says the tested number changes, and then test and code change together.
4. Light and dark screenshots at 1440 × 900 and 390 × 844 of every screen the stage touched,
   looked at, with no console errors.

Python is not touched in V0 to V10, so `uv run pytest` is run once at the end of each stage only
to confirm that (it must pass unchanged).

### V0 · Groundwork: tokens and stylesheet split

| | |
|---|---|
| **Objective** | Make the styles ready to change, without changing how anything looks. |
| **Components** | The stylesheet only. |
| **Files** | `web/src/styles.css` → `web/src/styles/tokens.css`, `base.css`, `environment.css`, `network.css`, `experiment.css`; `web/src/main.tsx` (imports). |
| **Work** | Split by the twelve existing section comments. Introduce the spacing, radius, type-size and duration tokens with *today's* values. Add `--pos` / `--neg` (aliases of today's `--chart-pos` / `--chart-neg`), `--w-*`, `--car-*`, `--ray-*`, `--track-*` tokens set to today's canvas colours. Move dark values under `:root[data-theme="dark"]` and keep a `prefers-color-scheme` block that sets the same values. |
| **Depends on** | Nothing. |
| **Visual improvement** | None. That is the test. |
| **Difficulty** | Medium: large but mechanical. |
| **Risks** | A rule lost or reordered in the split changes a detail somewhere. |
| **Verification** | The standing checks, plus: screenshots of every view, tool and layout taken before and after (light and dark) are pixel-identical, or differ only by anti-aliasing. |

### V1 · Identity: palette, type, shell

| | |
|---|---|
| **Objective** | The app gets its colours, its typeface and its frame. |
| **Recommendations** | G1 to G8, G10 (first set of shared controls), G11, G12. |
| **Files** | `web/src/styles/tokens.css`, `base.css`; `web/package.json` (three font packages); `web/src/main.tsx`; `web/src/components/TopBar.tsx`, `ConnectionBadge.tsx`, `CompanionDrawer.tsx`; new `web/src/components/Breadcrumb.tsx`, `web/src/components/ui/*`; `web/src/nav/views.ts`; `web/src/state/layout.ts` (+ `layout.test.ts`); `web/src/App.tsx`. |
| **Work** | Token values from `DESIGN_SPEC.md` §2 to §4. Reassign every use of `--accent`: selection stays, sign moves to `--pos`, status to its tokens. Fonts. Top bar, breadcrumb, theme switch (`theme` in the layout store, `data-theme` on `<html>`), drawer header. Buttons, segmented controls, chips and fields move to shared components view by view. |
| **Depends on** | V0. An ADR for the font dependency and for the theme setting, before starting. |
| **Visual improvement** | High: every screen changes character at once. |
| **Difficulty** | Medium. |
| **Risks** | Wider or narrower text wraps labels (risk 6 of the mapping). The canvas still reads page tokens until V2, so the track looks odd for one stage if `--surface` changes under it: give the `--w-*` tokens their V0 values until V2. |
| **Verification** | The standing checks. The palette run through the dataviz validator on the new surfaces (series, sign pair, status fills, both themes) with the output kept in the task report. Theme switch: all three positions, remembered after reload. Contrast of text tokens checked against `DESIGN_SPEC.md` §2.2. |

### V2 · The track

| | |
|---|---|
| **Objective** | The road becomes a place: asphalt, kerbs, sand, verge, grass, markings. |
| **Recommendations** | E1 to E4, E10. |
| **Files** | `web/src/views/environment/draw.ts` → `scene.ts`, `drawWorld.ts` (cars and overlays still in their old functions, moved to `drawCars.ts`, `drawOverlays.ts`); new `trackArt.ts` + `trackArt.test.ts`; `web/src/styles/tokens.css` (`--w-*` values); `web/src/views/experiment/DatasetView.tsx` only if its map is to share the surface (optional, can wait for V9). |
| **Work** | The world theme (`readTheme` reads `--w-*`). World drawn under a metres transform, in the layer order of `DESIGN_SPEC.md` §7.1. `trackArt` computes curvature, kerb runs, sand polygons, arrows, boards and the start line once per geometry. |
| **Depends on** | V0. |
| **Visual improvement** | The highest of any stage. |
| **Difficulty** | Medium to high: a rewrite of the drawing approach, not of its inputs. |
| **Risks** | Frame time (mapping risk 4). Kerbs or sand that fold over themselves on a track shape the prototype never saw: it was only run on `train_loop_01`. |
| **Verification** | The standing checks. `trackArt` unit tests: on a circle, constant curvature of 1 / radius within 1 %, an inside kerb all the way round and none outside; on every track in `tracks/`, no kerb or sand point lies inside the road (distance to the centreline at least half the width). Screenshots of **all six tracks** in both themes at whole-track and at follow zoom. Frame time of `drawScene` measured with 1, 20 and 100 cars before and after, on a performance and an efficiency core; the live view must stay inside 33 ms. Rays still end exactly on the white lines (zoom in and look). |

### V3 · Cars and what is drawn on the track

| | |
|---|---|
| **Objective** | Cars look like cars and say how they are doing; rays and trails are readable on any ground. |
| **Recommendations** | E5 to E9, E17. |
| **Files** | `web/src/views/environment/drawCars.ts`, `drawOverlays.ts`, `scene.ts`; `EnvironmentView.tsx` (legend markup only); `web/src/styles/tokens.css`. |
| **Work** | Sprite and dart, shadow, brake lights, night headlights; rank bodies, gold elite ring, dashed ghost; stopped cars faded with a badge per status; selection reticle and name tag; liveries for other drivers; ray fan, casing, pills; cased trail; what-if paths and ghost; look-back car. |
| **Depends on** | V2 (the metres transform). |
| **Visual improvement** | High. |
| **Difficulty** | Medium. |
| **Risks** | The enlarged dart at overview zoom hides a neighbour in a tight pack; name tags and ray labels overlapping. The rank ramp on asphalt must be re-validated. |
| **Verification** | The standing checks. Screenshots of every driver: keyboard, constant, neuron, random networks (20 and 100 cars), expert, checkpoint replay, GA generation; and of every car state: running, crashed, finished, stalled, out of time, selected, elite, ghost, look-back, what-if ghost. `population.spec.ts`, `ghost.spec.ts`, `trail.spec.ts`, `linking.spec.ts` unchanged and passing (they click cars and trail points by their reported positions). |

### V4 · The stage

| | |
|---|---|
| **Objective** | Nothing covers the track that does not have to; the controls used most are the easiest to reach. |
| **Recommendations** | E11 to E16, E18, G8 (Standings tab). |
| **Files** | `web/src/views/environment/Hud.tsx` (→ telemetry), `Controls.tsx` (→ `ScenarioBar.tsx`, `Transport.tsx`, `CameraBar.tsx`), `Trail.tsx`, `EnvironmentView.tsx`, `geometry.ts` (+ tests); new `Minimap.tsx`, `Standings.tsx`; `web/src/components/CompanionDrawer.tsx`; `web/src/state/layout.ts`; `web/src/styles/environment.css`. |
| **Work** | In this order, each of which can ship alone: (a) telemetry strip; (b) controls split and placed; (c) look-back strip; (d) minimap; (e) Standings; (f) camera easing, look-ahead and fit margins. |
| **Depends on** | V1 (shared controls), V3. |
| **Visual improvement** | High. |
| **Difficulty** | High: the most interactive code in the app, and the most tested. |
| **Risks** | Mapping risks 1 and 3. The record controls and the car picker lose their place if not planned (the mapping §3.2 gives each one). `hud-step` is read 48 times by tests: the strip must expose the same ids with the same text formats. |
| **Verification** | The standing checks; for (f), `environment-camera.spec.ts` and the click-a-car tests run 20 times in a row with no failure, and the camera tests' expected numbers changed in the same commit as the code. By hand: drive a lap with the keyboard; record, save and discard a demonstration; race a GA generation and change generation, start and ghost; look back by slider and by clicking the trail; what-if with the ghost. In Whole track view no floating panel overlaps the road on any of the six tracks. |

### V5 · Network diagram

| | |
|---|---|
| **Objective** | The diagram can be read: curves instead of a mesh, values without hovering, clear focus. |
| **Recommendations** | N1 to N6, N12, N13. |
| **Files** | `web/src/views/network/NetworkDiagram.tsx`, `networkLayout.ts` (+ `networkLayout.test.ts`), `diff.ts` (+ `diff.test.ts`), `web/src/styles/network.css`; `web/src/views/experiment/ChartHover.tsx` (reused for the read-out card). |
| **Depends on** | V1. |
| **Visual improvement** | High. |
| **Difficulty** | Medium. |
| **Risks** | Mapping risk 2 (`stroke-width` and sign classes are read by a test). Layout tests pin today's coordinates: thirteen expectations change with `layoutNodes`. |
| **Verification** | The standing checks. Screenshots for the layer sizes the project uses: [7, 2], [7, 8, 2], [7, 16, 16, 2], [7, 64, 64, 2] and four hidden layers (study S2a's shapes), in both themes, with nothing selected, a hidden neuron selected, an output selected, a connection selected, checkpoint change on, mutation on. Tooltip numbers equal the read-out line and equal `data-weight` / `data-activation`. |

### V6 · Network inspector rail

| | |
|---|---|
| **Objective** | The explanation sits beside the decision. |
| **Recommendations** | N7 to N11, N15, N17. |
| **Files** | `web/src/views/network/NetworkView.tsx`, `OutputInspector.tsx`, `WhatIfPanel.tsx`, `NeuronView.tsx`, `ParamSlider.tsx`, `NetworkCompact.tsx`; new `ActivationPlot.tsx`, `OutputCards.tsx`; `web/src/state/selection.ts` (+ tests) for the pointed-at neuron; `web/src/styles/network.css`. |
| **Depends on** | V5. |
| **Visual improvement** | Medium, with a large gain in use: no scrolling to see why. |
| **Difficulty** | Medium. |
| **Risks** | The rail's own scrolling hides the what-if panel on short screens; the neuron driver's view has a different shape and must not be forgotten. |
| **Verification** | The standing checks; `inspector.spec.ts`, `neuron.spec.ts` (11 tests), `ghost.spec.ts`, `lineage.spec.ts`, `timeline.spec.ts` unchanged. The activation plot's point equals the receipt's z and value for ten random frames (read from `data-*`). At 1280 × 720 the output cards, the receipt and the sum are visible without scrolling. |

### V7 · Chart kit

| | |
|---|---|
| **Objective** | Every chart is drawn sharp at its real size, with one anatomy and an objective badge. |
| **Recommendations** | X1 to X3. |
| **Files** | New `web/src/views/experiment/Plot.tsx`; `ChartHover.tsx`; then, one at a time: `LossChart.tsx`, `LiveChart.tsx`, `CompareChart.tsx`, `PredictionPlot.tsx`, `Evolution.tsx` (`GenerationFrame` and the seven charts in it), `Ancestry.tsx`, `DatasetView.tsx`, `ToyNetwork.tsx`; `web/src/styles/experiment.css`. `charts.ts` and its tests do not change. |
| **Depends on** | V1. |
| **Visual improvement** | Medium on each chart, on all of them. |
| **Difficulty** | Medium to high: many charts, each with hover maths in viewBox units to convert. |
| **Risks** | Hover positions off by the margin after the unit change; tests that hover at a fraction of a chart's box (`training.spec.ts`, `compare.spec.ts`, `evolution.spec.ts`, `toylab.spec.ts`). |
| **Verification** | The standing checks. Each converted chart at 400, 800 and 1600 px wide: tick text the same pixel size at all three; hovering the left edge, the middle and the right edge reads the first, middle and last unit. Converted one chart per commit, so a failing test names its chart. |

### V8 · Timeline dashboard

| | |
|---|---|
| **Objective** | A run reads top to bottom: what it is, how it went, how it does on the exam, then the detail. |
| **Recommendations** | X4 to X11. |
| **Files** | `web/src/views/experiment/Timeline.tsx`; `Evolution.tsx` split into `FitnessChart.tsx`, `DistributionChart.tsx`, `Histogram.tsx`, `DiversityChart.tsx`, `StartsMap.tsx`, `OutcomeGrid.tsx`; `Ancestry.tsx`; `evolutionCharts.ts`; new `ExamCard.tsx`; `web/src/styles/experiment.css`. |
| **Depends on** | V7. |
| **Visual improvement** | High. |
| **Difficulty** | Medium. |
| **Risks** | A run that is not a GA run (random search, imitation, DAgger) must still get a sensible page: summary, its own curve, the exam card, the per-car rows. Runs trained before tasks 7.6 to 7.8 lack generations, per-start results or lineage and must keep their "train again" messages. |
| **Verification** | The standing checks; `timeline.spec.ts`, `evolution.spec.ts`, `lineage.spec.ts`, `population.spec.ts` unchanged. Opened by hand: a GA run, a GA run trained before 7.7, a random-search run, an imitation run, a DAgger run, a baseline run with one card, a run still training. The status fills re-validated for colour-blind separation. |

### V9 · Compare, Training, Toy lab, Datasets

| | |
|---|---|
| **Objective** | The other four tools join the system. |
| **Recommendations** | X12 (grouping by method and sweep), X13, X16, X17, N18 (optional). |
| **Files** | `web/src/views/experiment/CompareRuns.tsx`, `RunList.tsx`, `CompareChart.tsx`, `TrainingRuns.tsx`, `LiveChart.tsx`, `ToyLab.tsx`, `LossChart.tsx`, `PredictionPlot.tsx`, `ToyNetwork.tsx`, `DatasetView.tsx`, `ExperimentView.tsx`. |
| **Depends on** | V7. |
| **Visual improvement** | Medium. |
| **Difficulty** | Medium. |
| **Risks** | The run list with about 270 runs must stay fast and findable; a filter must never hide a ticked run without saying so. |
| **Verification** | The standing checks; `compare.spec.ts`, `training.spec.ts`, `toylab.spec.ts`, `datasets.spec.ts` unchanged. A run launched, watched and stopped from the Training tool. A diverged run's chart shows no clipped label (audit X4). |

### V10 · Small windows and the All layout

| | |
|---|---|
| **Objective** | Each view is usable in whatever width it is given. |
| **Recommendations** | G9, N16, `DESIGN_SPEC.md` §12. |
| **Files** | `web/src/styles/*.css` (container queries), `web/src/views/network/NetworkView.tsx`, `web/src/views/environment/EnvironmentView.tsx`, `web/src/App.tsx` (the All layout's minimum panel sizes). |
| **Depends on** | V4, V6, V8. |
| **Visual improvement** | Medium; fixes audit R1 to R3. |
| **Difficulty** | Low to medium. |
| **Risks** | The `compact` prop and the width rule disagreeing in a drawer. |
| **Verification** | The standing checks. Every view at 390, 560, 768, 940, 1120, 1440 and 1920 px, in Page, Split and All: no horizontal scrolling of the page, no control cut off, the track always visible on the Environment page. |

### F1 · Flow mode and routes to the inputs (future)

| | |
|---|---|
| **Objective** | See what each connection carries now, and trace a decision back to the sensors through every layer. |
| **Recommendations** | N14. |
| **Files** | `src/neurodrive/server/schemas.py` (`Explanation`), `src/neurodrive/server/live.py`, `src/neurodrive/explain/contributions.py` (already computes them), `tests/server/test_live.py`, `tests/explain/`; `web/src/api/generated.ts` (regenerated); `docs/ARCHITECTURE.md` §7 and §8.2; `docs/DECISIONS.md` (ADR); then `NetworkDiagram.tsx`, `networkLayout.ts`. |
| **Depends on** | V5, and a decision by the user: this is an API change (v1.25 → v1.26), so it follows the project's rule: options, recommendation, ADR, then the task. |
| **Difficulty** | Medium. |
| **Risks** | Message size and frame rate with large networks ([7, 64, 64, 2] has 4,672 connections); mapping risk 5. |
| **Verification** | Python: for 200 random networks, each row of the sent contributions sums to z − b for its neuron to 1e-12 (the check task 2.9 already makes). Browser: a line's width follows the sent number; the routes lit for an output are the largest by the sent numbers. |

### F2 · Every method on the Driving Exam (future)

Belongs to **roadmap task 8.1** ("Comparison page"). When that task is reached, use
`DESIGN_SPEC.md` §8.5 (exam strip) and §8.3 (the filled badge) as its visual brief, and
`PRODUCTION_MAPPING.md` §6.2 for the data it needs. It is not a visual stage and should not be
built before Phase 8 decides the comparison's rules (budgets, seeds, which checkpoint counts).

---

## 3. What to verify at the very end

After the last stage taken:

1. The full standing checks, and `uv run pytest` unchanged.
2. Every function in `UI_AUDIT.md` §4 exercised once by hand, ticked off in a list.
3. One recorded run replayed before and after the whole redesign gives identical frames
   (the redesign must not have touched what is simulated or sent): compare two recordings of the
   same scenario and seed, as `tests/env/test_episode.py` does.
4. The screenshots in this folder's `screenshots/` compared side by side with the same screens of
   the app. Differences are either intended (listed in the stage's report) or bugs.
5. `docs/ARCHITECTURE.md` §8 updated to describe the shell, the breadcrumb, the theme setting and
   the drawer tabs as built.

---

## 4. How this fits the project's rules

- **One task at a time.** Each stage (and each lettered part of V4) is a task with a Brief and a
  Report in `task-notes/`, like any roadmap task. They would be numbered apart from the phases
  (for example `V2`, `V4c`) so `docs/CHECKLIST.md` can list them without disturbing phase
  numbers. Adding those rows is itself a change to the project's documents: the user decides
  whether and when.
- **Branches.** A visual stage is developed on its own branch from an up-to-date `main` (for
  example `visual-2-track`), not on a phase branch, so a phase's history stays about its phase.
- **Decisions first.** Three choices need an ADR before the work: the font dependency (V1), the
  theme setting in the layout store (V1), and the API addition (F1). The design itself, once
  approved, can be recorded as one ADR pointing at `DESIGN_SPEC.md`.
- **Contracts.** V0 to V10 change no contract of `ARCHITECTURE.md` §6 and no API. F1 changes the
  API by an addition. Nothing changes simulation, physics, learning or run folders.
- **The prototype stays a reference.** Its code is read, not copied wholesale: `mockServer.ts`,
  `store.ts`, `types.ts` and `Notes.tsx` must not be ported
  (`PRODUCTION_MAPPING.md` §5). The drawing code (`scene.ts`, `trackArt.ts`), the encodings
  (`encode.ts`), the chart kit (`charts.tsx`) and the stylesheets are the parts worth lifting.
