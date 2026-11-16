# Audit of the current NeuroDrive interface

What the app looks like and how it behaves today, before any redesign. Everything here was
checked against the code and against the running app, not only against the documents.

- **Version audited:** branch `phase-7-genetic-algorithm`, commit `13f6b61` (task 7.8), on
  2026-11-15.
- **How:** the documents listed in the brief were read; the frontend source under `web/src/` was
  read; the app was started with its normal commands (`uv run python -m neurodrive serve`,
  `npm run dev` in `web/`) and driven through its screens with Playwright at 1440 × 900 in light
  and dark, plus the All layout and a 390 px phone width. Nothing was changed.
- **Images:** `screenshots/before/` holds nine of the screens referred to below.

## 1. What the current interface does well

These are the things a redesign must keep.

1. **The linking works.** One selection (car, neuron, connection, output, moment) drives all three
   views, and it survives page changes because the stores belong to the app, not to a page
   (`web/src/state/selection.ts`, `playback.ts`, `live.ts`).
2. **Python is the only source of numbers.** The browser draws and sends commands. No view computes
   a network output or a physics step. This is the project's central rule and it is followed
   everywhere.
3. **Honest encodings.** Blue is positive and orange is negative in every view (weights,
   activations, contributions, sensitivity, the trail, the rays). The series palette and the rank
   ramp were checked with the dataviz validator. Every chart has a table view and a legend.
4. **Words beside numbers.** "−0.08 → slightly left", "If Ray −30° rose by 0.1, steer would change
   by about −0.056". The inspector reads like an explanation, not a dump.
5. **Careful interaction details.** Wheel zoom around the cursor, drag to pan, double-click to
   follow; thin lines have wide invisible hit areas; hover read-outs are HTML so they stay sharp;
   a click on the selected car returns to "now".
6. **Light and dark both exist** and both pass contrast checks.
7. **Everything is testable.** `data-testid` and `data-*` attributes on almost every element, 80
   browser tests in `web/e2e/`.
8. **Three layouts** (Page with a companion drawer, Split, All), remembered per browser.

## 2. Weaknesses

### 2.1 The whole app

| # | Finding | Evidence |
|---|---|---|
| A1 | **No visual identity.** System font, white cards on light grey, one blue. It reads as a default admin page, not as a racing simulator or a lab. | Any screenshot in `before/` |
| A2 | **One colour does four jobs.** `--accent` is the selected car, the primary button, a positive weight, and a positive sensitivity. On the Network page the selected output's ring, the positive lines and the pressed tab are all the same blue. | `web/src/styles.css` lines 7 and 13 (`--accent` and `--chart-pos` are both `#2563eb`) |
| A3 | **No hierarchy between controls and content.** Every view opens with rows of same-sized selects, number boxes and buttons. Play has the same weight as Zoom out. | `current-environment-population-light.png`: three rows of controls above the track |
| A4 | **Theme follows the system only.** There is no switch in the app; a user on a light system cannot see the dark theme. | `styles.css` line 37: the dark values exist only inside `@media (prefers-color-scheme: dark)` |
| A5 | **The selection is invisible as a whole.** The concept document's chain "Run → Checkpoint → Car → Neuron → Moment" is never shown in one place. The user infers it from what each view happens to display. | No component shows it |
| A6 | **Long instructions as body text.** The Environment page ends with a three-line paragraph of instructions that is always visible. | `EnvironmentView.tsx`, `env-hint` |
| A7 | **One stylesheet of 2,271 lines** with about 350 rules and no spacing or type scale. Sizes are written ad hoc (`0.92rem`, `0.85rem`, `0.8rem`, `11px`). | `web/src/styles.css` |
| A8 | **Titles carry scaffolding.** "Environment (compact)", "Experiment: run timeline". | View headers |

### 2.2 Environment view

| # | Finding | Evidence |
|---|---|---|
| E1 | **The track is a diagram, not a place.** A white ribbon on grey, two grey outlines, a dashed centre line and one black line for the start. No surface, no surroundings, no depth. | `draw.ts` `drawRoad`; `current-environment-keyboard-light.png` |
| E2 | **The canvas borrows the page's colours.** Road = `--surface`, background = `--bg`, edges = `--text-muted`. The track therefore inverts between themes (white road by day, near-black road by night) and cannot have its own palette. | `draw.ts` `readTheme` |
| E3 | **Cars are plain rectangles.** Direction is a thicker front edge. Unselected cars are grey, the selected one blue, a stopped one red: crashed, stalled, finished and out of time all look the same on the track. | `draw.ts` `drawCars` |
| E4 | **The HUD covers the track.** It is a table of up to twelve rows over the upper left. In population mode it is about 470 × 310 px and hides a third of the circuit in Whole track view; the legend then covers the lower right. | `current-environment-population-light.png` |
| E5 | **Ray labels are bare text** drawn straight on the canvas in the page's text colour, with no backing, so they collide with edges and with each other. | `draw.ts` `drawRays` |
| E6 | **The trail is thin and uncased**; on a white road an orange 1.5 px segment is hard to see. | `draw.ts` `drawTrail` |
| E7 | **Camera changes jump.** Follow ↔ Whole track and a change of selected car cut instantly. Following centres the car, so half the screen shows road already driven. | `EnvironmentView.tsx` `draw` |
| E8 | **No overview while following.** To see where the car is on the lap the user must leave follow mode. | — |
| E9 | **Population mode has no ranking list.** Rank is only a body colour; to find "the third best car" the user opens a drop-down of 51 entries. | `Controls.tsx` `select-car` |
| E10 | **Look back is a bare slider.** Nothing shows where in the last 60 s anything interesting happened. | `Trail.tsx` |
| E11 | **Status "Running" is shown while paused at step 0.** The chip means "not ended", which is correct, but beside a Play button it reads as "is playing". | `Hud.tsx`, `status.ts` |

### 2.3 Network view

| # | Finding | Evidence |
|---|---|---|
| N1 | **400 straight lines form a mesh.** Between two layers of 16 the lines cross into a moiré; individual weights are hard to follow to their ends. | `current-network-population-dark.png` |
| N2 | **The inspector is below the diagram.** On a 900 px screen it starts under the fold, so the explanation of a decision is not visible together with the diagram, while the right 40 % of the page is empty. | `current-network-inspector-light.png`: the page is 1,900 px tall |
| N3 | **Outputs are two small circles** with a number beside them. The thing the user must click to open the inspector is the least prominent element; a hint paragraph has to say so. | `NetworkView.tsx` `inspection-hint` |
| N4 | **Line meaning changes through checkboxes in banners.** "Colour lines by change" and "Colour lines by mutation" are in two different banners, and which one applies depends on the driver. | `NetworkView.tsx` |
| N5 | **Bias and weighted sum are only in the tooltip text.** The step from z to the output through the activation function is a line of text; nothing shows saturation. | `OutputInspector.tsx` `inspector-sum` |
| N6 | **"Strongest paths to inputs" is not built.** The architecture document lists it for a selected output; the code highlights only the lines that touch the selected neuron. | `ARCHITECTURE.md` §8.2 vs `NetworkDiagram.tsx` `isRelated` |

### 2.4 Experiment view

| # | Finding | Evidence |
|---|---|---|
| X1 | **The Timeline tool is one column, about 2,600 px tall** for a GA run: controls, a chart, four small charts, two grids, a chart, three collapsed tables, a table. There is no summary at the top and no grouping. | `current-experiment-timeline-dark.png` |
| X2 | **The checkpoint slider scrolls away.** The slider that drives the whole page is at the top; by the ancestry chart it is out of reach. | Same image |
| X3 | **Charts scale their text.** Charts are drawn in a fixed `viewBox` and stretched, so the first chart's tick labels are about 25 px tall at 1440 px while the small charts' labels are about 8 px. About fifteen charts in nine files are drawn this way. | `Timeline.tsx`, `Evolution.tsx`; same image |
| X4 | **Clipped axis labels.** With a diverged run the loss chart's left labels are cut off ("0e+260"). | `current-layout-all-dark.png`, right panel |
| X5 | **Nothing on a chart says which objective it shows.** "best fitness after each unit" and "Loss after each epoch" are told apart only by their titles. The concept document insists that loss, fitness and reward are not interchangeable. | All chart titles |
| X6 | **The Driving Exam has no special standing.** It is the only common yardstick, yet in the Timeline it is the last table at the bottom of the page. | `Timeline.tsx` `timeline-card` |
| X7 | **Compare runs lists every run in the folder** as one flat list of check boxes (about 270 runs on the audited machine) with no search, no filter by method and no grouping by study. Ticking runs of a method without a loss gives a correct but empty page. | `current-experiment-compare-light.png` |
| X8 | **Comparing methods is not possible yet**, which is by design (roadmap task 8.1), but the page gives no hint of where that comparison will live. | `docs/ROADMAP.md` Phase 8 |
| X9 | **Training progress is text only**: "unit 3 of 30". There is no bar and no sign of life between units. | `TrainingRuns.tsx` |
| X10 | **Five tools as a segmented control inside the view header** compete with the Split layout's view picker for the same row. | `ExperimentView.tsx` |

### 2.5 Small windows

| # | Finding | Evidence |
|---|---|---|
| R1 | **The All layout is not usable at 1440 px.** In a 480 px panel the Environment controls take six rows and the HUD plus the legend cover the entire canvas; the network diagram's labels are about 5 px tall. | `current-layout-all-dark.png` |
| R2 | **On a 390 px phone the track cannot be seen.** In population mode the controls fill the whole screen, one or two per row; the canvas is below the fold, and the open drawer takes 45 % of the width. | `current-phone-environment-light.png` |
| R3 | **The Experiment page's drawer** shows a very tall, mostly empty strip of track. | `current-experiment-compare-light.png`, right |

## 3. Where the documents and the code disagree

The code is what is implemented. These are the differences found; none was "fixed" by assuming.

| # | Document says | Code does | Consequence for the redesign |
|---|---|---|---|
| D1 | `README.md` "Status": Phase 6 complete, "Next: Phase 7". | `docs/CHECKLIST.md` and the code: tasks 7.0 to 7.8 are verified; population mode, evolution charts and lineage exist. | The redesign covers Phase 7's screens. The README is stale. |
| D2 | `ARCHITECTURE.md` §8.1 lists the selection store's fields as `mode`, `runId`, `unit`, `carId`, `neuron`, `connection`, `step`, `output`, `whatIf`. | `selection.ts` declares `mode`, `runId`, `unit` and `step` but nothing reads or writes them. The replayed run and checkpoint live in the live store (`scenario.checkpoint`, `scenario.population`); the moment lives in `playback.ts` (`viewStep`). The document's own "as built" notes say this. | The breadcrumb must read three stores, not one. Removing the four dead fields is optional tidying. |
| D3 | §8.2: clicking an output neuron should "highlight strongest paths to inputs" in the Network view and "show the output's response curve across checkpoints" in the Experiment view. | Only the lines touching the selected neuron are emphasised. The response curve does not exist (`POST explain/response` is listed as planned). | The prototype shows the strongest paths; in production they need data Python does not send yet (see `PRODUCTION_MAPPING.md` §6). |
| D4 | §8.2: clicking a car should "mark its run/checkpoint" in the Experiment view. | True only inside the Timeline tool during a replay (the card's row is highlighted). | The breadcrumb makes the run and generation visible everywhere. |
| D5 | The brief speaks of "existing light and dark themes". | Both exist as CSS, but only the operating system can choose. | A theme switch is new state (Category 2). |
| D6 | `web/src/nav/views.ts` gives every view a `comingIn` text and `components/ViewPanel.tsx` renders a "Placeholder" card. | All three views are built; the placeholder branch can no longer be reached for any view or drawer. | Dead code; safe to remove during a shell refactor. |
| D7 | §8.0 says the compact network is "a summary built for a narrow column, not a shrunken diagram". | True for the drawer. In the All layout the *full* diagram is shrunk into a 450 px panel instead. | The redesign switches by the view's own width, not by which slot it is in. |
| D8 | `CLAUDE.md` §6 lists `npm run e2e` as starting both servers. | True, and it also trains runs into `runs/` on first use. | Visual work must not change any `data-testid`; the tests are the safety net. |

## 4. What exists today and must survive

The redesign may move and restyle these; it may not remove them. The list is taken from the code.

**Environment:** track and driver selection (keyboard, constant, neuron, random networks, expert,
checkpoint, GA generation); number of cars (1 to 100); random start with a seed; play, pause, step,
reset; speeds 0.5× to 8×; keyboard driving (arrows or W A S D); recording keyboard demonstrations
(Record, tag, Save, Discard, the REC badge); car selection by click, by drop-down; follow, whole
track and free camera, wheel zoom, drag, double-click; rays with distances, emphasised rays for a
selected input, rays coloured by sensitivity; decision trail, coloured by a selected output; look
back by slider or by clicking the trail, with the past shown in the HUD; what-if ghost with its
legend and gap; population mode (rank colours, elite rings, faded stopped cars, ghost, generation /
start / ghost controls, race legend, role and fitness in the HUD); status banner and server error
banner; the compact view with its Follow button.

**Network:** the neuron driver's view (weights, bias, activation, throttle sliders, activation
curve); the multilayer diagram with hover and selection of neurons and connections; input and
output read-outs; car stepper and picker; replay banner with "colour lines by change"; lineage
banner with "colour lines by mutation"; the past banner with "Back to now"; output inspector
(tabs, headline, receipt, sum, sensitivity, the nudge sentence); what-if panel (sliders, reset,
now / what-if / predicted table, the paused and past hints); the compact network.

**Experiment:** Toy lab (run picker, checkpoint slider, loss chart, prediction plot or map,
gradient glow network); Compare runs (run list by sweep, training or validation loss, bands,
medians, per-run lines, crosshair, table, skipped notice); Training (launch form, run list, stop,
live chart with metric picker, exam table); Timeline (run and track pickers, start shift, watch
the expert, checkpoint slider, replay, race generation, the chart with ringed checkpoints, the
exam card rows, and for GA runs the evolution charts, start picker, starts map, outcome grid,
ancestry with its row picker, and the tables); Datasets (picker, compare with, label and
observation histograms with dense coverage, track map with pan and zoom, episodes).

**Shell:** three pages with hash routes, Page / Split / All, resizable panels, companion drawers,
the connection badge and reconnection.
