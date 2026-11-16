# Production mapping

Where each part of the prototype's design would go in the real NeuroDrive app, which existing
files are involved, and how much work each part is. It is written for whoever implements an
approved design later. File paths are real paths in this repository as of commit `13f6b61`; checked
again at `1e27c85` (no file under `web/` changed in between)
(task 7.8). Nothing in this document has been done; it is a map.

Contents: 1 Categories · 2 Every recommendation, classified · 3 File by file · 4 Shared state ·
5 New and reusable components · 6 What touches Python or a contract · 7 What must be preserved ·
8 Refactoring the frontend needs first · 9 Risks · 10 Since the live source

---

## 1. The three categories

| Category | Meaning | Test for it |
|---|---|---|
| **1 · Current architecture** | Can be built inside the frontend as it is: the same stores, the same messages, the same component boundaries. | Only files under `web/src/views/` and `web/src/styles.css` change, and no component gains or loses a job. |
| **2 · Frontend refactoring** | Needs frontend structure to change: a component split or moved, a store field added, the stylesheet reorganised, a rendering approach replaced. Nothing in Python changes. | A component's responsibility, a store's shape or the styling architecture changes. |
| **3 · Future enhancement** | Needs something that does not exist yet: data Python does not send, an endpoint that is only planned, or a roadmap task that has not been done. | `src/neurodrive/`, `docs/ARCHITECTURE.md` §6 or §7, or `docs/ROADMAP.md` would have to change. |

A recommendation's category is that of its hardest part. Where a recommendation splits cleanly
(part now, part later) both parts are listed.

---

## 2. Every recommendation, classified

### 2.1 Whole app

| # | Recommendation | Cat. | Why this category |
|---|---|---|---|
| G1 | Design tokens: new palette, type scale, spacing, radii, shadows, motion (`DESIGN_SPEC.md` §2 to §4, §11) | 2 | The values are CSS only, but today's `styles.css` has no scales: about 350 rules with ad hoc sizes must be moved onto tokens. That is a stylesheet reorganisation, not a restyle. |
| G2 | Separate colours for selection (violet), sign (blue / orange) and status | 2 | `--accent` is used for all of them in CSS and is read by the canvas (`draw.ts` `readTheme`). Each use must be reassigned to `--accent`, `--pos` or a status token. |
| G3 | IBM Plex fonts, bundled | 2 | Three new npm packages in `web/package.json` and imports in `web/src/main.tsx`. A dependency decision: record an ADR first (CLAUDE.md §3.6). No runtime network use. |
| G4 | Dark top bar with brand mark, tabs, right-hand controls | 1 | `web/src/components/TopBar.tsx` and CSS. Same props. |
| G5 | Selection breadcrumb in the top bar | 1 | A new read-only component. It reads `useLive` (`scenario.checkpoint`, `scenario.population`, `frame.selected.index`), `useSelection` (`neuron`, `connection`) and `usePlayback` (`viewStep`). It writes nothing; each crumb is a link to a view. |
| G6 | Theme switch: light, dark, follow the system | 2 | New state (a `theme` field, best placed in `web/src/state/layout.ts`, which is already saved to local storage), a `data-theme` attribute on `<html>`, and the dark values moved from the `@media (prefers-color-scheme: dark)` block to an attribute selector. `draw.ts` and `DatasetView.tsx` read CSS variables per draw, so the canvases follow. |
| G7 | View header with the view's question; scenario and tool controls in it | 1 | Each view already renders its own `.view-header`. |
| G8 | Drawer as a panel with a header; Environment drawer with Network and Standings tabs | 1 / 2 | Restyle: 1 (`CompanionDrawer.tsx`, CSS). The tab choice is new layout state (`drawerTab` in `layout.ts`): 2. |
| G9 | Views adapt to their own width (container queries) | 2 | Replaces the `compact` prop as the only switch for the Network view; fixes the All layout (audit R1). CSS plus small changes in `NetworkView.tsx`. |
| G10 | Shared controls: buttons, segmented control, fields, chips, panels, stats, value bars, legends, tooltips | 2 | Today each view styles its own `<button>`, `<select>` and bar classes (`.bar`, `.cnet-*`, `.inspector-*`). Introducing shared components touches every view. |
| G11 | Loading skeletons and empty states with one action | 1 | Replaces the `.hint` paragraphs in place. |
| G12 | Reduced-motion support | 1 | One CSS block. |

### 2.2 Environment view

| # | Recommendation | Cat. | Why this category |
|---|---|---|---|
| E1 | Track surface: grass with stripes, verge, sand traps, asphalt with grain, rubber line, kerbs, edge lines, centre dashes, arrows, checkered start | 1 | All inside `web/src/views/environment/draw.ts` `drawRoad`, from the geometry `GET tracks/{name}` already returns. No new data. |
| E2 | A world palette of its own (`--w-*`), dark asphalt in both themes | 2 | `readTheme` maps the canvas to page tokens (`road = --surface`). The `Theme` interface gets a new shape and every draw function's colours are reassigned. |
| E3 | Decorative geometry computed once per track (`trackArt.ts`) | 1 | A new pure module next to `geometry.ts`, cached per `TrackGeometry` object. Unit-testable like `geometry.ts`. |
| E4 | Drawing the world in metres (a canvas transform) and labels in pixels | 2 | `draw.ts` converts every point with `worldToScreen`. Kerbs, textures and car sprites are far simpler under a world transform. This is a rewrite of how `draw.ts` draws, with the same inputs (`Scene`). |
| E5 | Car sprite, dart at overview, shadow, brake lights, headlights at night | 1 | Replaces `carPath` / `drawCars` / `drawRace`. Uses `frame.cars` fields that exist (`action` for brake lights). |
| E6 | Car states: stopped cars faded with a status badge; selected car's reticle and name tag; elite ring in gold; ghost dashed | 1 | `frame.cars.status` already distinguishes crashed, finished, stalled and timeout; today all are one red fill. |
| E7 | Liveries for non-population drivers (up to ten cars) | 1 | Drawing only. Not in the prototype (`DESIGN_SPEC.md` §7.2). |
| E8 | Rays: fan, casing, fade-in, hit dots, label pills pushed outward, dashed no-hit rays | 1 | `drawRays`. Same inputs. |
| E9 | Trail: cased; plain trail in the selection colour | 1 | `drawTrail`. |
| E10 | Distance boards every 100 m | 1 | From `geometry.length` and `start_s0`. |
| E11 | Telemetry strip replacing the HUD table | 2 | `Hud.tsx` is rebuilt as a different component with the same data and **the same `data-testid`s** (§7). It moves from the top left of the stage to the bottom. |
| E12 | Transport and camera controls floating on the stage; scenario fields in the header | 2 | `Controls.tsx` (427 lines, one component) is split into three: scenario, transport, camera. The record controls and the car picker need a place (§3.2). |
| E13 | Minimap | 1 | New component; `geometry` + `frame.cars`. |
| E14 | Standings in the drawer | 1 | New component; `scenario.population` (ranks, fitness, elite) + `frame.cars` (progress, status). For non-population drivers it lists cars by progress without rank or F. |
| E15 | Look-back strip drawn on the steer and throttle history | 1 | `Trail.tsx`. `live.trail.action` holds every remembered step's action already. |
| E16 | Camera: eased transitions; look-ahead while following; fit margins that keep clear of the floating panels | 1 | `EnvironmentView.tsx` `draw` and two new pure functions in `geometry.ts`. **Changes numbers that tests assert** (§9, risk 3). |
| E17 | Status banner and legend as stage panels | 1 | Same content, restyled; the legend stays click-through. |
| E18 | Long instruction paragraph replaced by tooltips and one hint | 1 | `EnvironmentView.tsx` `env-hint`. |
| E19 | Heading-up camera, replays of other cars' rays, weather or time-of-day beyond the two themes | 3 | Not proposed for now; listed so they are not mistaken for omissions. |

### 2.3 Network view

| # | Recommendation | Cat. | Why this category |
|---|---|---|---|
| N1 | Curved connections | 1 | `NetworkDiagram.tsx` `Edges`: `<line>` becomes `<path>`. Keep `data-testid`, `data-weight`, the `net-edge-pos` / `net-edge-neg` classes and the `stroke-width` attribute (§9, risk 2). |
| N2 | Layout: shared row spacing, wider spacing for outputs, larger labels | 1 | `networkLayout.ts` `layoutNodes` and `FRAME`; `networkLayout.test.ts` has 13 tests of the current positions to update. |
| N3 | Inputs with value bars; outputs with value, signed bar and words | 1 | `NetworkDiagram.tsx`; values are in `trace.activations`. |
| N4 | Column heads with activation function and parameter counts | 1 | From `network.activations`, `network.weights`. |
| N5 | Hover: floating read-out card; read-out line kept | 1 | `ChartHover.tsx` already exists for charts and can be reused. |
| N6 | Selected neuron: accent ring and halo; unrelated lines fade | 1 | CSS and `isRelated`. |
| N7 | Inspector as a rail beside the diagram | 2 | `NetworkView.tsx` `MultilayerView` / `Inspection` and CSS: the page becomes a two-column grid with its own scroll areas. `OutputInspector.tsx` and `WhatIfPanel.tsx` keep their logic. |
| N8 | Output cards at the top of the rail | 1 | Replaces `inspector-tabs` (keep the test ids `inspect-steer`, `inspect-throttle`). |
| N9 | Activation curve with the neuron's point on it | 1 | `useLive(s => s.activations)` already holds Python's curves (`GET activations`); `NeuronView.tsx` draws one today and `curve.ts` has the helpers. |
| N10 | Inspector rows light the neuron they name | 2 | A transient "pointed at" value shared between `OutputInspector.tsx` and `NetworkDiagram.tsx`: a new field in `selection.ts` (the prototype's `peek`) or a small context. |
| N11 | "Lines show" as one switch (Weights / Mutation / Change) | 2 | Replaces two module-level booleans in `NetworkView.tsx` (`showChangeChoice`, `showMutationChoice`) and two checkboxes in banners with one value. Which options are enabled depends on the driver, as today. Keep `show-change` and `show-mutation` as test ids on the options. The prototype's live source now has all three, Change from `GET .../diff` in a checkpoint replay, with production's own `changeStroke` and `mutationStroke`. |
| N12 | Lineage sentence with parents' ranks and gene counts | 1 | `diff.ts` `originText`; `MutationView` already has `parents_rank`, `n_from_other`, `n_mutated`. |
| N13 | Strongest contributors into a selected output stay lit (one layer back) | 1 | `explain.receipts[m].contributions` is in every frame. |
| N14 | Strongest *routes* back to the inputs (all layers), and the **Flow** line mode | 3 | Needs weight × value for every connection of every layer. Python computes it (`src/neurodrive/explain/contributions.py`) but the server sends receipts for the two outputs only. See §6.1. With the live source the prototype disables Flow and lights the routes one layer back from the receipts (N13); only its recorded stand-in has every layer. |
| N15 | What-if: locked states say why and offer the fix | 1 | `WhatIfPanel.tsx` already has both hints; add the button. |
| N16 | Narrow view: the compact summary instead of the diagram | 2 | With G9. `NetworkCompact.tsx` is reused as it is. |
| N17 | Neuron driver view (`NeuronView.tsx`) on the new components | 1 | Not in the prototype. Same content: sliders become the shared slider rows, the curve is N9's plot. |
| N18 | Gradient glow as a fifth "Lines show" mode in the Toy lab | 2 | `ToyNetwork.tsx` is a second, separate diagram today; merging it into `NetworkDiagram.tsx` is a refactor. Optional. |

### 2.4 Experiment view

| # | Recommendation | Cat. | Why this category |
|---|---|---|---|
| X1 | Charts drawn at their real size (no scaled text) | 2 | About fifteen charts in nine files use a fixed `viewBox` stretched by CSS. Each needs a measured width. The pure helpers in `charts.ts` stay. One shared `Plot` wrapper does it for all (§8.3). |
| X2 | Chart anatomy: hairline grid, tick and label styles, line and band weights, capped bars, ringed markers | 1 | CSS classes `.toy-*`, `.evolution-*` and attribute values. |
| X3 | Objective badge on every chart | 1 | A small component in every `<figcaption>`; the chart already knows what it plots. |
| X4 | Timeline as a dashboard grid with a run summary on top | 2 | `Timeline.tsx` and `Evolution.tsx` (933 lines, one file with eight charts) are re-laid-out; `Evolution.tsx` is worth splitting into one file per chart. |
| X5 | Sticky generation bar; one cursor for all charts | 2 | The slider (`timeline-slider`) and the Evolution panel's generation are already linked (`panelGeneration`); they move into one bar. |
| X6 | Middle-half band on the fitness chart | 1 | `EvolutionView` already has `q1` and `q3`. |
| X7 | Checkpoint rings on the fitness chart | 1 | `Timeline.tsx` already draws ringed dots on its own chart from `GET runs/{id}/checkpoints`; the two charts merge. |
| X8 | Heat map sharing the fitness chart's generation axis | 1 | Both use `generationScale`; they are side by side today and become stacked. |
| X9 | Driving Exam card with training and held-out blocks and a per-track table | 1 | `GET runs/{id}/exams/{unit}` returns the whole card; today only the chosen track's ten rows are shown. The per-car rows (and clicking one to select the car) stay, under the card. |
| X10 | Outcome grid: colour-blind-safe status fills, outlines for the raced start and the selected car | 1 | `OUTCOMES` in `evolutionCharts.ts` and CSS tokens. |
| X11 | Ancestry in the accent colour over a dashed reference | 1 | `Ancestry.tsx`, CSS. |
| X12 | Compare: run list grouped and filterable by method and study | 2 | `RunList.tsx` and `CompareRuns.tsx`. `RunSummary` has `method`, `tag`, `sweep`; `meta.json` has `study` (ADR-043) but `GET runs` does not return it: grouping by study needs that one field added (a small, additive API change: 3). Grouping by method and sweep is 2. |
| X13 | Compare: labels at line ends, table restyled | 1 | `CompareChart.tsx` already places end labels. |
| X14 | **Every method on the Driving Exam** (the strip chart) | 3 | The exam cards exist in every run folder, but nothing gathers them across runs and seeds, and medians over seeds are statistics, which belong in Python (`experiment/stats.py`). This is roadmap task 8.1 ("Comparison page"); `ARCHITECTURE.md` §7 lists `POST compare` as planned. |
| X15 | Compare fitness curves of several runs (bands of best fitness) | 3 | `GET runs/compare` accepts `metric=loss` or `val_loss` only. |
| X16 | Training: progress bars, pulsing last point, shaded not-yet-reached axis, stats row | 1 | `TrainingRuns.tsx`, `LiveChart.tsx`; `run_status` messages carry `units_done` and `budget_units`. |
| X17 | Toy lab and Datasets on the chart kit | 1 / 2 | Restyle: 1. Real-size drawing: with X1. |
| X18 | Response curve of an output across checkpoints | 3 | `POST explain/response` is planned, not built (`ARCHITECTURE.md` §7, §8.3 item 4). Not designed here. |

---

## 3. File by file

### 3.1 Shell

| Production file | What changes | Prototype reference |
|---|---|---|
| `web/src/styles.css` | Split (§8.1). Tokens replaced by `DESIGN_SPEC.md` §2 to §4. | `src/theme/tokens.css`, `base.css` |
| `web/src/main.tsx` | Font imports. | `src/main.tsx` |
| `web/src/App.tsx` | The three layouts stay. `PageLayout` passes the drawer its tab. | `src/App.tsx` |
| `web/src/components/TopBar.tsx` | Brand mark, tabs with icons, breadcrumb slot, theme switch; keeps the layout switch (Page / Split / All) and `ConnectionBadge`. | `TopBar`, `Crumbs` in `src/App.tsx` |
| `web/src/components/ConnectionBadge.tsx` | Restyle; keep `data-testid="connection-badge"` and `data-status`. | `.conn` |
| `web/src/components/CompanionDrawer.tsx` | Header, collapsed rail, tabs on the Environment page. The Experiment page's companion stays the Environment view. | `Drawer` in `src/App.tsx` |
| `web/src/components/ViewPanel.tsx`, `comingIn` in `web/src/nav/views.ts` | Unreachable today (audit D6). Remove, or keep for future views. | — |
| `web/src/nav/views.ts` | Add each view's question and icon. | `VIEWS` in `src/App.tsx` |
| `web/src/state/layout.ts` | Add `theme` and `drawerTab`; extend `sanitizeLayout` and its tests (`layout.test.ts`). | `store.ts` |
| new `web/src/components/Breadcrumb.tsx` | G5. | `Crumbs` |
| new `web/src/components/ui/` | Shared controls (§5). | `src/ui/` |

### 3.2 Environment

| Production file | What changes | Prototype reference |
|---|---|---|
| `web/src/views/environment/draw.ts` | Rewritten in layers (E1, E2, E4 to E10). The `Scene` interface keeps its fields; `Theme` is replaced. Suggested split: `scene.ts` (the order of layers), `drawWorld.ts`, `drawCars.ts`, `drawOverlays.ts`. | `scene.ts` |
| new `web/src/views/environment/trackArt.ts` (+ test) | E3. | `trackArt.ts` |
| `web/src/views/environment/geometry.ts` | Add `easeCamera`, `lookAhead`; `fitCamera` takes a margin per side. Existing functions and `geometry.test.ts` stay valid. | `camera.ts` |
| `web/src/views/environment/EnvironmentView.tsx` | Camera easing and look-ahead in `draw`; the stage's children become: transport, camera bar, minimap, banner, legend, telemetry. All `data-*` attributes on the canvas are kept. | `Stage.tsx`, `EnvironmentView.tsx` |
| `web/src/views/environment/Hud.tsx` | Becomes the telemetry strip (E11). The compact drawer version stays a one-line read-out. | `Telemetry.tsx` |
| `web/src/views/environment/Controls.tsx` | Split into `ScenarioBar`, `Transport`, `CameraBar` (E12). `IntField` and `ghostAfterMove` are reused unchanged. Placement of what the prototype does not show: **Cars, Random start, Seed** in the scenario bar; **Selected car** picker in the scenario bar (the Standings tab and the minimap also select); **Record, REC badge, Tag, Save, Discard** as a second floating bar under the transport bar, shown for the keyboard driver only; **Start shift** stays in the Timeline tool. | `EnvironmentView.tsx` |
| `web/src/views/environment/Trail.tsx` | E15. Keeps the slider's test id and the paused-only rule. | `TrailStrip` in `Extras.tsx` |
| `web/src/views/environment/population.ts` | Unchanged (`carRole`, `roleLabel`, band maths). | `rankBand` in `Stage.tsx` |
| `web/src/views/environment/status.ts`, `trailLayout.ts`, `keys.ts`, `useDrivingKeys.ts` | Unchanged. | — |
| new `Minimap.tsx`, `Standings.tsx` | E13, E14. | `Extras.tsx` |

### 3.3 Network

| Production file | What changes | Prototype reference |
|---|---|---|
| `web/src/views/network/NetworkDiagram.tsx` | N1, N3 to N6, N13. `Edges` stays memoised. The three legend components merge into one that follows "Lines show". | `Diagram.tsx`, `Legend` in `NetworkView.tsx` |
| `web/src/views/network/networkLayout.ts` | N2: `layoutNodes`, `FRAME`; add `edgePath`. `edgeWidth`, `edgeOpacity`, `nodeFill`, the labels and descriptions stay. | `encode.ts` |
| `web/src/views/network/diff.ts` | `changeStroke`, `mutationStroke`, `originText` keep their maths; `originText` gains the ranks and counts (N12). | `encode.ts`, `originText` in `NetworkView.tsx` |
| `web/src/views/network/NetworkView.tsx` | N7, N11, N16: two-column body, "Lines show", banners restyled. `useCheckpointChange`, `useMutation` and `CarStepper` keep their logic. | `NetworkView.tsx` |
| `web/src/views/network/OutputInspector.tsx` | N8, N9, N10: cards, receipt rows, the sum with the curve, sensitivity rows. `inspect.ts` helpers (`rankedReceipt`, `topAndRest`, `nudgeSentence`) are reused. | `Inspector.tsx` |
| `web/src/views/network/WhatIfPanel.tsx` | N15, restyle. | `WhatIf` in `Inspector.tsx` |
| `web/src/views/network/NetworkCompact.tsx` | Restyle; reused for N16. `ValueBar` moves to the shared controls. | `NetworkCompact.tsx` |
| `web/src/views/network/NeuronView.tsx`, `ParamSlider.tsx`, `curve.ts` | N17, restyle. | — |

### 3.4 Experiment

| Production file | What changes | Prototype reference |
|---|---|---|
| `web/src/views/experiment/ExperimentView.tsx` | Tool switch with icons; order unchanged or as the prototype (Timeline first). | `ExperimentView.tsx` |
| `web/src/views/experiment/charts.ts` | Unchanged helpers; add nothing that computes statistics. | `charts.tsx` (the pure half) |
| new `web/src/views/experiment/Plot.tsx` | X1: measured-width SVG, `YAxis`, `XAxis`, `Figure`, `Objective`. | `charts.tsx` (the components) |
| `web/src/views/experiment/ChartHover.tsx` | Becomes the one read-out card for charts and the diagram. | `Tip`, `TipRow` |
| `web/src/views/experiment/Timeline.tsx` | X4, X5, X7, X9: run summary, generation bar, exam card. Start shift and "Watch the expert" stay as a row in the run summary. | `RunTimeline.tsx` |
| `web/src/views/experiment/Evolution.tsx` | X2, X4, X6, X8, X10. Suggested split into `FitnessChart.tsx`, `DistributionChart.tsx`, `Histogram.tsx`, `DiversityChart.tsx`, `StartsMap.tsx`, `OutcomeGrid.tsx`; `GenerationFrame` becomes the shared cursor. The three collapsed tables stay. | `RunTimeline.tsx` |
| `web/src/views/experiment/evolutionCharts.ts` | `OUTCOMES` colours point at the new fill tokens; shading helpers unchanged. | — |
| `web/src/views/experiment/Ancestry.tsx`, `ancestryChart.ts` | X11. The row picker stays. | `AncestryChart` |
| `web/src/views/experiment/CompareRuns.tsx`, `RunList.tsx`, `CompareChart.tsx` | X12, X13. The training / validation switch, per-run lines, crosshair, skipped notice stay. | `Compare.tsx` |
| `web/src/views/experiment/TrainingRuns.tsx`, `LiveChart.tsx` | X16. | `Training.tsx` |
| `web/src/views/experiment/ToyLab.tsx`, `LossChart.tsx`, `PredictionPlot.tsx`, `ToyNetwork.tsx` | X17 (`DESIGN_SPEC.md` §8.7). | — |
| `web/src/views/experiment/DatasetView.tsx`, `datasetMap.ts` | X17. Its track map can reuse `drawWorld.ts` (§3.2) for the surface. | — |

---

## 4. Shared state

The prototype keeps one store of its own (`src/state/store.ts`) for layout, view options and the
recorded clock. With the live source its selection and clock are production's own `useSelection`,
`usePlayback` and `useLive`, through a bridge in `src/source/live.ts`; the table maps the
prototype's fields to them. Production keeps its four stores.

| Prototype field | Production home today | Change proposed |
|---|---|---|
| `data` (recorded files) | `useLive`: `scenario`, `geometry`, `frame`, `network`, `trail`, `moment`, `whatIfResult`, `episodeEnd`; REST for runs | None |
| `t`, `playing`, `speed` | The server's clock: `frame.step`, `frame.playing`, `frame.speed` | None. The prototype's interpolation between frames has no production counterpart and should not get one: the server sends a frame at most every 1/30 s. |
| `viewKey` | `usePlayback.viewStep` | None |
| `car` | `useSelection.carId` (+ `pendingCar`) | None |
| `neuron`, `connection`, `output`, `whatIf` | `useSelection`, same names | None |
| `peek` | — | **New**, transient, in `useSelection` (N10). Not part of the selection contract: it is cleared on mouse leave and never sent. |
| `view` | `web/src/nav/route.ts` (the hash) | None |
| `layout`, `drawerOpen` | `useLayout.mode`, `.drawerOpen` (per view), `.splitPartner` | None |
| `drawerTab` | — | **New** in `useLayout` (G8) |
| `theme` | — | **New** in `useLayout` (G6); values `light`, `dark`, `auto` |
| `cameraMode` | Component state in `EnvironmentView.tsx` (`mode`) | None needed. Lift it into `useLayout` only if the camera bar is rendered outside `EnvironmentView`. |
| `showRays`, `showTrail` | — | **New**, component state in `EnvironmentView.tsx`, passed into the `Scene` |
| `lineMode` | Two module-level booleans in `NetworkView.tsx` | **Replace** with one value kept the same way (N11) |
| `experimentTab` | Component state in `ExperimentView.tsx` | None. (Optional: put it in the hash, `#/experiment/timeline`, so a tool can be linked to. `parseHash` already ignores anything after the view name.) |
| `generation`, `evoStart` | Component state in `Timeline.tsx` / `Evolution.tsx` | None |
| `notes` | — | Prototype only. Do not port. |

`useSelection`'s unused fields `mode`, `runId`, `unit` and `step` (audit D2) are not needed by
the breadcrumb, which reads the live scenario. They can be deleted in the same change, together
with the matching lines of `ARCHITECTURE.md` §8.1; that is a documentation fix, not a contract
change.

---

## 5. New and reusable components

**New** (suggested location `web/src/components/ui/` unless noted):

| Component | From the prototype | Used by |
|---|---|---|
| `Icon` | `src/ui/Icon.tsx` | Everything |
| `Button`, `Segmented`, `Field`, `Select`, `Chip`, `Panel`, `Stat`, `ValueBar`, `LegendItem`, `Swatch` | `src/ui/controls.tsx` | Everything |
| `Breadcrumb` | `Crumbs` in `src/App.tsx` | `TopBar` |
| `Plot`, `YAxis`, `XAxis`, `Figure`, `Objective` (`web/src/views/experiment/`) | `charts.tsx` | Every chart |
| `Minimap`, `Standings`, `Telemetry`, `Transport`, `CameraBar`, `ScenarioBar` (`web/src/views/environment/`) | `Extras.tsx`, `Telemetry.tsx`, `EnvironmentView.tsx` | Environment |
| `ActivationPlot`, `OutputCards` (`web/src/views/network/`) | `Inspector.tsx` | Network |
| `ExamCard` (`web/src/views/experiment/`) | `ExamPanel` in `RunTimeline.tsx` | Timeline, Training |

**Reused as they are** (logic untouched, at most restyled):

`web/src/api/*` · `web/src/state/live.ts`, `selection.ts`, `playback.ts`, `runsWatch.ts` ·
`web/src/nav/route.ts` · `geometry.ts` · `population.ts` · `status.ts` · `trailLayout.ts` ·
`keys.ts` · `useDrivingKeys.ts` · `inspect.ts` · `diff.ts` · `curve.ts` · `charts.ts` ·
`evolutionCharts.ts` · `ancestryChart.ts` · `datasetMap.ts` · `ChartHover.tsx` ·
`NetworkCompact.tsx` · `IntField` · `CarStepper` · `useCheckpointChange` · `useMutation`.

**Do not port:** `src/source/recorded.ts`, `startLive` in `src/source/live.ts` (the bridge that lets
one component run on two sources), `src/mock/mockServer.ts`, `src/mock/types.ts`, `src/state/store.ts`,
`src/ui/Notes.tsx`, `scripts/`. They exist so the prototype can run alone.

---

## 6. What touches Python or a contract

Three things in the whole design. Everything else is display.

### 6.1 Contributions for every layer (N14: Flow, and routes back to the inputs)

- **Today:** each frame's `selected.explain` holds `receipts` for the two outputs
  (`OutputReceipt`: `inputs`, `contributions`, `bias`, `z`, `value`) and `sensitivity`
  (`src/neurodrive/server/schemas.py`, classes `OutputReceipt` and `Explanation`).
- **Needed:** for the selected car, c = weight × source value for every connection of every
  layer, each frame, and the same in `moment` and `what_if` answers.
  `src/neurodrive/explain/contributions.py` already computes contributions for any neuron
  (task 2.9: "contributions sum to z − b for every neuron").
- **Size:** for a 7-16-16-2 network, 400 numbers per frame for one car on top of the 75 numbers of
  the trace. It can be sent only while a "flow wanted" flag is set, as rays' sensitivity is tied
  to a selected output today.
- **Process:** an addition to the server API (`ARCHITECTURE.md` §7): an ADR, `API_VERSION`
  v1.25 → v1.26, `npm run gen:types`, server tests. No change to §6 contracts (observation,
  action, network file, recording).
- **Why not in the browser:** the multiplication is trivial, but "explanation maths lives in
  Python" is a non-negotiable rule (CLAUDE.md §4). The prototype does it in `mockServer.ts` and
  marks the mode "proposed".
- **Without it:** N13 still works (the largest contributors into the selected output, one layer
  back), from the receipts that exist.

### 6.2 The exam of every method on one chart (X14)

- **Today:** `GET runs/{id}/exams/{unit}` returns one card. Nothing returns the last card of every
  seed of several methods, or their medians.
- **Needed:** an endpoint that, for a list of runs or a study, returns per group the per-seed
  training and held-out aggregates and their median and quartiles. `ARCHITECTURE.md` §7 already
  lists `POST compare {run_ids} → aligned exam and budget tables` as planned; roadmap task 8.1
  is the comparison page.
- **Process:** part of Phase 8, with its own ADR. The design here is input to that task, not a
  replacement for it. The prototype built the chart from 43 separate card requests at capture
  time and computed quartiles in the browser, which production must not do.

### 6.3 Small additions that would help, none required

| Addition | For | Without it |
|---|---|---|
| `study` in each `GET runs` entry | Grouping the run list by study (X12) | Group by method and sweep only |
| `metric=best_fitness` on `GET runs/compare` | Bands of fitness across seeds (X15) | Compare stays loss-only, as today |
| Per-car `status` is sent as a number; nothing needed | Badges (E6) | — |

### 6.4 What needs no change, checked

Track decoration (the geometry message is enough) · car states (`frame.cars.status`, `.alive`,
`.action`) · rank colours, elites, ghost (`scenario.population`) · standings (`frame.cars.progress`,
`population.fitness`) · the look-back strip (`trail.action`) · the activation curve
(`GET activations`) · the fitness band (`EvolutionView.q1`, `.q3`) · the exam card (the full
card is already returned) · lineage sentence (`MutationView`) · what-if and its ghost
(`what_if`, `GhostPaths`) · look back (`inspect_step`, `moment`).

No versioned contract of `ARCHITECTURE.md` §6 changes. Simulation, physics, learning, run folders
and reproducibility are untouched by every item in this document.

---

## 7. What must be preserved

1. **Every function listed in `UI_AUDIT.md` §4.** The prototype leaves several out (keyboard
   driving, recording, the neuron driver, checkpoint replay with shifted starts, the expert, Toy
   lab, Datasets, the All layout); production keeps them all.
2. **The linking rules of `ARCHITECTURE.md` §8.2**, including: `selectCar` is the only way to
   change the car; the store follows the server's `selected.index`; a new scenario clears neuron,
   connection and output; what-if and look-back are dropped when the step or car changes; a click
   on the selected car returns to now.
3. **Mouse behaviour on the canvas** (task 4.10): wheel zooms the level while following and around
   the cursor otherwise; a drag of 4 px or more pans and frees the camera; double-click follows;
   a click within 24 px selects a car, within 12 px a trail point (paused only).
4. **Test hooks.** The 80 browser tests find elements by `data-testid` and read `data-*`
   attributes. All of these must survive a restyle with the same meaning:
   - canvas: `environment-canvas` with `data-cars`, `data-camera`, `data-camera-mode`,
     `data-trail`, `data-trail-values`, `data-moment`, `data-highlighted-rays`,
     `data-ray-sensitivity`, `data-ghost`, `data-race`;
   - HUD: `hud`, `hud-speed`, `hud-status`, `hud-steer`, `hud-throttle`, `hud-step` (used 48
     times), `hud-progress`, `hud-past`, `hud-rays`, `ray-0` … `ray-4`, `hud-race`,
     `hud-race-role`, `hud-race-fitness`, `hud-race-start-fitness`, `status-banner`,
     `server-error`;
   - controls: `select-track`, `select-policy`, `input-cars`, `check-random-start`, `input-seed`,
     `btn-play`, `btn-step`, `btn-reset`, `btn-record` and the record ids, `select-speed`,
     `select-car`, `camera-follow`, `camera-fit`, `camera-free`, `zoom-in`, `zoom-out`,
     `race-generation`, `race-start`, `race-ghost`;
   - legends: `race-legend`, `ghost-legend` (with `ghost-gap`), `ray-legend`;
   - network: `network-diagram`, `net-node-L-I` (`data-activation`, `data-selected`),
     `net-edge-L-F-T` (`data-weight`, `data-change`, `data-mutated`, `data-from-other`,
     the `stroke-width` attribute and the `net-edge-pos` / `net-edge-neg` class), `net-hit-L-F-T`,
     `net-input-K`, `net-output-K`, `network-readout`, the three legend ids, `car-prev`,
     `car-next`, `car-pick`, `replay-banner`, `show-change`, `lineage-banner`, `show-mutation`,
     `past-banner`, `past-live`, the inspector and what-if ids, the `cnet-*` ids;
   - experiment: every id in `Timeline.tsx`, `Evolution.tsx`, `Ancestry.tsx`, `CompareRuns.tsx`,
     `CompareChart.tsx`, `TrainingRuns.tsx`, `LiveChart.tsx`, `ToyLab.tsx`, `DatasetView.tsx`,
     `RunList.tsx`.
   The prototype reuses many of these names but was not built against the tests.
5. **Python as the only source of numbers.** No ported component may compute a network output, a
   rollout, a statistic or an explanation.
6. **Validated palettes.** Any colour that encodes data is re-run through the dataviz validator on
   the surface it is drawn on before it ships, as tasks 3.8, 4.7b and 7.6 did.

---

## 8. Refactoring the frontend needs first

Three pieces of groundwork make the rest cheap. None changes behaviour.

### 8.1 Split the stylesheet and introduce tokens

`web/src/styles.css` (2,271 lines) into: `tokens.css` (all custom properties, light and dark
under `:root[data-theme]`), `base.css` (reset, shell, shared controls), and one file per view.
Then replace literal sizes and colours with tokens, section by section (the file already has
twelve named sections). Until G6 lands, keep a `@media (prefers-color-scheme: dark)` block that
sets the same dark values, so nothing changes for the user.

### 8.2 Split `draw.ts` and give the canvas its own theme

Separate "what colours" (`readTheme` → a world theme read from `--w-*`, `--car-*`, `--ray-*`,
`--track-*`) from "what to draw", and split drawing into world, data layers, cars and labels
(the order in `DESIGN_SPEC.md` §7.1). `drawScene(ctx, theme, scene)` keeps its signature, so
`EnvironmentView.tsx` does not change in this step. A first version can keep today's look with
the new structure; the new look then arrives layer by layer.

### 8.3 One chart frame

`Evolution.tsx` already has `GenerationFrame`; `LossChart`, `LiveChart`, `CompareChart`,
`PredictionPlot`, `Ancestry` and `DatasetView` each set up their own `viewBox`, margins and axes.
A shared `Plot` (measures its width, renders at 1:1) and shared axis components replace seven
copies. `pointerX` and the hover maths switch from viewBox units to pixels, which removes a class
of bugs (the stretched read-outs of tasks 4.7b and 5.8).

---

## 9. Risks

| # | Risk | Where | How to keep it small |
|---|---|---|---|
| 1 | **Browser tests break on moved or renamed elements.** | Everywhere | Keep every `data-testid` (§7.4). Run `npm run e2e` after each stage, not at the end. |
| 2 | **`network.spec.ts` reads each edge's `stroke-width` and sign class** and compares them with the weight; `toylab.spec.ts` reads `stroke-opacity`. | `web/e2e/network.spec.ts:58`, `toylab.spec.ts:79` | Keep `edgeWidth` and `glowStroke` as the source of the attribute, or change formula and test together. A `<path>` carries `stroke-width` as well as a `<line>` does. |
| 3 | **Camera look-ahead and new fit margins change `data-camera`.** `environment-camera.spec.ts` asserts zoom factors and that a world point stays under the cursor (unaffected), but tests that click a car at its `data-cars` position rely on the camera being settled; an eased transition makes "settled" later. | `web/e2e/environment-camera.spec.ts`, `linking.spec.ts`, `ghost.spec.ts`, `trail.spec.ts` | Write `data-cars` from the camera actually used for the frame (the prototype does). Add a `data-camera-settled` attribute and have tests wait on it. Skip easing when `prefers-reduced-motion` is set and run the tests with it set. |
| 4 | **Canvas cost.** The new world has more strokes, two pattern fills and up to 100 car sprites. | `draw.ts` | The prototype draws 51 cars every display frame and looked smooth on the development machine, but its frame time was **not measured**. Measure before and after; if needed: render the static world to an offscreen canvas and redraw it only when the camera moves. The live view's budget is 33 ms per frame (`ARCHITECTURE.md` §9). |
| 5 | **Flow mode re-renders 400 paths per frame.** | `NetworkDiagram.tsx` `Edges` | `Edges` is memoised on the network today; in Flow mode it would depend on the trace. Update only `stroke-width` and `stroke-opacity`, or cap at 15 updates a second. |
| 6 | **Fonts change text widths** and can wrap labels that fit today. | All views | Land fonts with the token stage, review every view at 1280 px and 390 px. |
| 7 | **Scope creep into behaviour.** A visual stage quietly changing what a control does. | All | One stage per branch; each stage's verification includes the unchanged Vitest and Playwright suites. |
| 8 | **Roadmap collision.** Phase 8's comparison page (task 8.1) and this design's exam strip are the same screen. | `docs/ROADMAP.md` | Treat `DESIGN_SPEC.md` §8.5 and X14 as the visual brief for task 8.1 when it is reached; do not build it earlier as a visual stage. |

---

## 10. Since the live source (task VP, 2026-11-16)

The prototype now runs on the real server through production's own client code. What that settled
is in `MIGRATION_AUDIT.md`: what it reuses (§2), what must not move (§3), the duplicates and how
each is resolved (§4), what was proved on live data (§5), the test hooks it keeps (§6), the three
production refactors the migration needs first (§7) and the findings in production code (§8).
`IMPLEMENTATION_ROADMAP.md` is the step-by-step plan built on both documents.
