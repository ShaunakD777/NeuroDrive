# NeuroDrive 2D redesign prototype

A design prototype: what NeuroDrive's three views could look like, built as a small, separate web
app that can be opened and clicked through. It is a visual reference, not a new version of the
app.

- It lives only in this folder. Nothing in `web/`, `src/`, `docs/`, the root configs or the
  roadmap was changed to make it.
- It never talks to the Python server. It replays one race and one run that were recorded from the
  real server once and saved as plain JSON files.
- It does no simulation and no learning. The few sums it does to stand in for Python are in one
  file, `src/mock/mockServer.ts`, which says so at the top.

## The documents

| File | What it is for |
|---|---|
| `README.md` | This file: what the prototype is and how to run it |
| `UI_AUDIT.md` | The current app as it is today: what works, what does not, and where the docs and the code disagree |
| `DESIGN_SPEC.md` | The design system: colours with exact values, type, spacing, components, charts, track and cars, themes, motion, small screens |
| `PRODUCTION_MAPPING.md` | Where each part of the design would go in the real app: actual files, what can be reused, what must stay, and a category (1, 2 or 3) for every recommendation |
| `IMPLEMENTATION_ROADMAP.md` | A staged plan for bringing an approved design into the real app, separate from `docs/ROADMAP.md` |
| `screenshots/` | Images of the running prototype (`screenshots/before/` holds the current app for comparison) |

## Run it

You need Node.js with npm (tested with Node 24, the same as the main app).

```
cd prototypes/2d-redesign
npm install
npm run dev
```

Then open http://127.0.0.1:5180. The port is different from the main app's (5173) and the Python
server's (8000), so all three can run at once.

| Command | What it does |
|---|---|
| `npm run dev` | Start the prototype on port 5180 |
| `npm run build` | Type-check, then build into `dist/` |
| `npm run typecheck` | Type-check only |
| `npm run screenshots` | Retake every image in `screenshots/` (start `npm run dev` first; the first time, run `npx playwright install chromium` if the browser is missing) |
| `npm run capture` | Record the data again from a running NeuroDrive server (see "Recorded data") |

## What to try

1. **Environment.** Press Play (or Space). Scroll to zoom, drag to pan, double-click to follow the
   car again. Click another car. Switch to Whole track. Open the Standings tab in the drawer.
2. **Network.** Click Steer or Throttle in the rail on the right: the strongest routes to that
   output stay lit, the inspector explains the value, and the rays on the track (drawer) are
   coloured by sensitivity. Hover a neuron or a line. Switch "Lines show" to Flow or Mutation.
3. **What-if.** Pause, select an output, move a slider under "What if". Go back to the Environment
   view: a ghost car shows where the changed action would take the car.
4. **Look back.** Pause, then drag on the strip under the track, or click a point of the trail.
5. **Experiment.** Drag the generation slider or click any chart. Click a cell of the outcome grid
   to select that car everywhere. Open Compare runs and Training.
6. **Theme and layout.** Top right: sun, moon, or follow the system; Page or Split.
7. **Notes.** Top right: numbered markers appear on the screen and the drawer explains each one,
   with the category of work it would take in the real app.

## Structure

```
prototypes/2d-redesign/
├── index.html, package.json, tsconfig.json, vite.config.ts
├── public/
│   ├── favicon.svg
│   └── recorded/            the recorded data (JSON), see below
├── scripts/
│   ├── capture-mock-data.mjs   records public/recorded/ from a running server
│   └── screenshots.mjs         retakes screenshots/
├── screenshots/             images of the prototype; before/ = the current app
└── src/
    ├── main.tsx, App.tsx    entry point; shell (top bar, breadcrumb, drawer, layouts, theme)
    ├── theme/
    │   ├── tokens.css       every colour, size and timing, light and dark
    │   ├── base.css         shell and shared controls
    │   └── environment.css, network.css, experiment.css
    ├── ui/                  Icon, shared controls, the design-notes layer
    ├── state/store.ts       one store: selection, playback, layout, options
    ├── mock/
    │   ├── types.ts         shapes of the recorded files (mirror web/src/api/generated.ts)
    │   └── mockServer.ts    the stand-in for Python
    └── views/
        ├── environment/     scene.ts (canvas drawing), trackArt.ts (kerbs, sand, boards),
        │                    camera.ts, Stage.tsx, Telemetry.tsx, Extras.tsx, EnvironmentView.tsx
        ├── network/         encode.ts, Diagram.tsx, Inspector.tsx, NetworkCompact.tsx, NetworkView.tsx
        └── experiment/      charts.tsx (the chart kit), RunTimeline.tsx, Compare.tsx, Training.tsx,
                             ExperimentView.tsx
```

The folder the task description calls `assets/` is not used: the prototype has no image files.
The track, the cars and the icons are all drawn by code, and the fonts come from npm packages.

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

These are copies. The prototype reads nothing from `runs/` or `data/`. To record again, start the
real server (`uv run python -m neurodrive serve` in the repository root) and run `npm run capture`
here. The script only reads from the server; options are at the top of
`scripts/capture-mock-data.mjs`.

The folders are named `public/recorded/` and `src/mock/`, not `data`, because the repository's
`.gitignore` ignores every folder named `data`, and this exercise does not change that file.

## Limits

- **One scenario.** One track, one generation, one start. The track, driver, generation, start and
  ghost fields are shown but fixed.
- **Five networks.** Any car can be selected on the track, but only the five recorded cars have a
  network to show. For the others the rays are measured by the prototype from the track edges.
- **Frames are 8 steps apart.** Positions between them are interpolated; Step moves one recorded
  frame (8 simulation steps), and pausing lands on a recorded frame so every number on screen
  belongs to one real step.
- **Stand-in maths.** The what-if answer, the ghost path, the per-connection "flow" and the tanh
  curve are computed in `mockServer.ts` and `Inspector.tsx`. In the real app all of these come
  from Python; two of them need Python to send more than it does today (see
  `PRODUCTION_MAPPING.md` §6).
- **Three of five Experiment tools.** Timeline, Compare runs and Training are built. Toy lab and
  Datasets are described in `DESIGN_SPEC.md` §8.7 and show a short note in the prototype.
- **Other drivers.** Keyboard driving, the neuron driver with its sliders, recording
  demonstrations, checkpoint replay with shifted starts and the expert are not in the prototype.
  They keep their functions; `PRODUCTION_MAPPING.md` §3 says how each takes on the new look.
- **Layouts.** Page and Split are built. The app's All layout and resizable panels stay as they are.
- **Not tested like the app.** No unit tests, no browser tests. `npm run build` passes and
  `npm run screenshots` runs with no console errors; that is the whole check.

## Dependencies

The prototype has its own `package.json` and lock file. It adds nothing to the main app.

- Runtime: `react`, `react-dom`, `zustand` (the same as the main app), and three font packages,
  `@fontsource/ibm-plex-sans`, `@fontsource/ibm-plex-sans-condensed`, `@fontsource/ibm-plex-mono`
  (SIL Open Font License; the fonts are bundled, nothing is fetched from the internet at runtime).
- Development: `vite`, `@vitejs/plugin-react`, `typescript`, `playwright` (for the screenshots).
