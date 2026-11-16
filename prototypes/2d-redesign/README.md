# NeuroDrive 2D redesign prototype

A design prototype: what NeuroDrive's three views could look like, built as a small, separate web
app that can be opened and clicked through. It is a working reference for the redesign, not a new
version of the app.

- It lives only in this folder. Nothing in `web/`, `src/`, `docs/`, `tests/` or the root configs
  was changed to make it.
- It has **two sources of data**, chosen in the top bar:
  - **Recorded**: one GA race and one run, recorded once from the real server and saved as JSON.
    Works without any server.
  - **Live**: the real Python server (`nd serve`), through the production app's own API client and
    stores, imported read-only from `web/src/`. Every driver, track and run folder.
- It does no simulation and no learning. With the live source it computes nothing either: every
  number is Python's. The few sums the recorded source needs to stand in for Python are in one
  file, `src/mock/mockServer.ts`, which says so at the top.

## The documents

| File | What it is for |
|---|---|
| `README.md` | This file: what the prototype is and how to run it |
| `UI_AUDIT.md` | The current app as it is today: what works, what does not, and where the docs and the code disagree |
| `DESIGN_SPEC.md` | The design system: colours with exact values, type, spacing, components, charts, track and cars, themes, motion, small screens |
| `MIGRATION_AUDIT.md` | The prototype measured against the production code: what it reuses, duplicates and must not take, which test hooks it keeps, what the live source proved |
| `PRODUCTION_MAPPING.md` | Where each part of the design goes in the real app: files, reuse, what must stay, a category (1, 2 or 3) per recommendation |
| `IMPLEMENTATION_ROADMAP.md` | The migration plan: view by view, each step with its files, its tests and how it is undone |
| `screenshots/` | Images of the running prototype; `live/` with the real server; `before/` the current app |

## Run it

You need Node.js with npm (tested with Node 24, the same as the main app), and for the live
source the repository's Python setup (`uv sync` in the repository root).

```
cd prototypes/2d-redesign
npm install
npm run dev
```

Then open http://127.0.0.1:5180. The page starts with the recorded source.

For the live source, start the Python server in the repository root first:

```
uv run python -m neurodrive serve
```

then click **Recorded** in the top bar (it becomes **Live**), or open
http://127.0.0.1:5180/?source=live. The choice is remembered per browser. The dev server forwards
`/api` and `/ws` to port 8000 exactly as `web/` does, so the real app (5173), the prototype (5180)
and the server (8000) can run at the same time.

| Command | What it does |
|---|---|
| `npm run dev` | Start the prototype on port 5180 |
| `npm run build` | Type-check (including the production files it imports), then build into `dist/` |
| `npm run typecheck` | Type-check only |
| `npm run screenshots` | Retake the recorded images in `screenshots/` and fail on any console error (start `npm run dev` first; the first time, run `npx playwright install chromium` if the browser is missing) |
| `npm run live-check` | Drive every driver and tool against the running server, fail on any failed check, console error or failed request, and retake `screenshots/live/` (`-- --no-shots` to skip the images). It loads scenarios and asks what-if questions; it trains nothing |
| `npm run capture` | Record the data again from a running NeuroDrive server (see "Recorded data") |

## What to try

With either source:

1. **Environment.** Press Play (or Space). Scroll to zoom, drag to pan, double-click to follow the
   car again. Click another car, or pick it in the Car field. Switch to Whole track. Open the
   Standings tab in the drawer.
2. **Network.** Click Steer or Throttle in the rail on the right: the strongest routes to that
   output stay lit, the inspector explains the value, and the rays on the track are coloured by
   sensitivity. Hover a neuron or a line. Switch "Lines show".
3. **What-if.** Pause, select an output, move a slider under "What if". Go back to the Environment
   view: a ghost car shows where the changed action would take the car.
4. **Look back.** Pause, then drag on the strip under the track, or click a point of the trail.
5. **Experiment.** Drag the generation slider or click any chart. Click a cell of the outcome grid.
   Open Compare runs and Training.
6. **Theme and layout.** Top right: sun, moon, or follow the system; Page or Split.
7. **Notes.** Top right: numbered markers appear on the screen and the drawer explains each one,
   with the category of work it would take in the real app.

Only with the live source:

8. **Every driver.** Driver: Keyboard (drive with the arrow keys or W A S D, record a
   demonstration), Constant, Neuron (its panel in the Network view sets the weights), Random
   networks (Cars: 1 to 100, Random start, Seed), Expert, GA generation (Run, Generation, Start,
   Ghost gen). Track: all six.
9. **Your runs.** Experiment → Timeline lists every driving run in `runs/`. A GA run opens as the
   dashboard; its generation panels follow the cursor. Race generation g, Replay champion, or
   click an outcome cell. Other methods show their metric, checkpoints and exam card. In a
   checkpoint replay the Network view's "Change" mode shows what learning changed.
10. **Compare and Training.** Tick a sweep; one task per chart. Training lists runs as they train
    and can start and stop them (this starts a real `nd train` process).

## Structure

```
prototypes/2d-redesign/
├── index.html, package.json, tsconfig.json, vite.config.ts   (@prod → ../../web/src, read-only)
├── public/recorded/          the recorded data (JSON), see below
├── scripts/
│   ├── capture-mock-data.mjs records public/recorded/ from a running server
│   ├── screenshots.mjs       retakes screenshots/ (recorded source)
│   └── live-check.mjs        checks the live source against a running server
├── screenshots/              images; live/ = the real server; before/ = the current app
└── src/
    ├── main.tsx, App.tsx     entry point; shell (top bar, breadcrumb, source switch, drawer, layouts, theme)
    ├── source/               the seam between the views and their data
    │   ├── model.ts          the world snapshot the views read (production message types)
    │   ├── recorded.ts       snapshot from public/recorded/ (uses src/mock/)
    │   ├── live.ts           snapshot from production's useLive / useSelection / usePlayback
    │   ├── world.ts          useWorld() / getWorld(): one snapshot, rebuilt only when an input changes
    │   └── runs.ts           the Experiment view's data: run list, /ws/runs, cached answers
    ├── state/store.ts        layout and view options; the recorded clock; the bridge to production's stores
    ├── mock/                 the recorded files' shapes and the stand-in for Python (recorded source only)
    ├── theme/                tokens.css (every colour, size and timing), base.css, one file per view
    ├── ui/                   Icon, shared controls, SourceWaiting, the design-notes layer
    └── views/
        ├── environment/      scene.ts (canvas), trackArt.ts, camera.ts, cars.ts (bodies, liveries),
        │                     Stage, Telemetry, Extras (minimap, legend, look-back, standings), EnvironmentView
        ├── network/          encode.ts, Diagram, Inspector, NeuronPanel, NetworkCompact, NetworkView
        └── experiment/       charts.tsx (the chart kit), RunTimeline (charts and dashboards),
                              Timeline (both sources), Compare, Training, ExperimentView
```

How the live source works, in one paragraph: `App.tsx` runs production's `useLiveConnection`
(the socket, reconnects, track outlines). `source/live.ts` hands the prototype's actions
(select a car, play, look back, what-if …) to production's stores, and keeps the prototype's
selection fields a copy of `useSelection`. `source/world.ts` rebuilds the snapshot from
`useLive`, `usePlayback` and `useSelection` whenever one of them changes. The views read only the
snapshot, so the same component draws the recording and the server.

## Recorded data

`public/recorded/` holds real data from the project's own runs, recorded on 2026-11-15:

| File | Contents |
|---|---|
| `track.json` | `train_loop_01` as the server sends it (`GET tracks/{name}`) |
| `race.json` | Generation 18 of the GA run `20261114-140528-ga-lineage78`, raced from start 1 with generation 17's champion as the ghost: 274 frames of 51 cars (every 8th simulation step) |
| `cars/<n>.json` | For five cars, the network and, per frame, the sensors, the trace and the explanation: the champion (46), the best finishing child (14), a car that crashes (30), one that runs out of time (26) and the ghost (50) |
| `run.json` | The run's evolution charts, generation 18's outcome grid and ancestry, two exam cards, one checkpoint diff |
| `compare.json` | Two comparisons from `GET runs/compare`: four optimizers (study S3) and three imitation teachers (S5c), thinned to about 150 points per curve |
| `exams.json` | The last exam card of every seed of seven methods |

These are copies. The recorded source reads nothing from `runs/` or `data/`. To record again,
start the real server and run `npm run capture` here. The script only reads from the server;
options are at the top of `scripts/capture-mock-data.mjs`.

The folders are named `public/recorded/` and `src/mock/`, not `data`, because the repository's
`.gitignore` ignores every folder named `data`.

## Limits

- **Recorded: one scenario.** One track, one generation, one start; the scenario fields are shown
  but fixed. Five cars have a network; for the others the stand-in measures the rays. Frames are
  8 steps apart and are interpolated while playing.
- **Live: what the server does not send yet.** "Flow" and routes back to the inputs need every
  layer's weight × value (an API addition, `PRODUCTION_MAPPING.md` §6.1); live, Flow is disabled
  and the strongest routes are lit one layer back, from Python's receipts. The exam of every
  method on one chart needs roadmap task 8.1; live, that panel says so.
- **Two of five Experiment tools are not built.** Toy lab and Datasets are described in
  `DESIGN_SPEC.md` §8.7 and show a short note. Checkpoint replay with shifted starts and
  "Watch the expert" stay in production's Timeline.
- **Layouts.** Page and Split are built. The app's All layout and resizable panels stay as they are.
- **Tested by scripts, not by the app's suites.** `npm run build` passes, `npm run screenshots`
  runs with no console errors, and `npm run live-check` passes against the server. The app's
  Vitest and Playwright suites do not run against the prototype; `MIGRATION_AUDIT.md` §6 measures
  how many of their test hooks it already keeps.

## Dependencies

The prototype has its own `package.json` and lock file. It adds nothing to the main app.

- Runtime: `react`, `react-dom`, `zustand` (the same as the main app, deduplicated so the imported
  production stores use the same copies), and three font packages, `@fontsource/ibm-plex-sans`,
  `@fontsource/ibm-plex-sans-condensed`, `@fontsource/ibm-plex-mono` (SIL Open Font License; the
  fonts are bundled, nothing is fetched from the internet at runtime).
- Development: `vite`, `@vitejs/plugin-react`, `typescript`, `playwright` (for the scripts).
- From `web/src/` (read-only, through the `@prod` alias): the generated message types, the REST and
  socket client, the live, selection, playback and runs-watch stores, and a few pure helpers
  (`MIGRATION_AUDIT.md` §2). The prototype therefore builds only inside this repository.
