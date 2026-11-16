# Screenshots

Every image except those in `before/` is the running prototype (Chromium through Playwright).
The images in this folder replay the recorded race (`npm run screenshots`); the images in `live/`
show the real Python server's simulation and run folders (`npm run live-check`). Desktop images
are 1440 × 900 unless the content is taller; phone images are 390 × 844.

## Environment

| File | Shows |
|---|---|
| `environment-follow-light.png`, `-dark.png` | The camera following the champion: day and night track, rays with distances, telemetry, minimap, compact network in the drawer |
| `environment-whole-track-standings-light.png`, `-dark.png` | Whole track late in the race (finished, crashed and running cars), Standings in the drawer |
| `environment-steer-selected-light.png` | Steer selected: rays coloured by sensitivity, trail coloured by the decision |
| `environment-look-back-dark.png` | Looking back at an earlier step: the violet car, outlined telemetry, "Back to now" |
| `environment-crashed-car-light.png` | A crashed car selected: badge, banner, status chip |
| `environment-ghost-selected-dark.png` | The ghost (last generation's champion) selected |
| `environment-what-if-ghost-light.png` | What-if: the real path (dashed), the what-if path and the ghost car |

## Network

| File | Shows |
|---|---|
| `network-weights-light.png`, `-dark.png` | The diagram with nothing selected |
| `network-steer-explained-light.png`, `-dark.png` | Steer selected: strongest routes lit, the inspector rail (why, sensitivity) |
| `network-flow-dark.png` | "Lines show: Flow" (proposed) |
| `network-mutation-light.png` | "Lines show: Mutation" for a child of two parents |
| `network-hover-neuron-light.png` | Hovering a hidden neuron: the read-out card, related lines lit |
| `network-neuron-selected-dark.png` | A hidden neuron kept selected |
| `network-what-if-light.png` | The what-if panel with two inputs changed |

## Experiment

| File | Shows |
|---|---|
| `experiment-timeline-light.png`, `-dark.png` | The GA run dashboard, whole page |
| `experiment-chart-hover-dark.png` | The read-out on the fitness chart |
| `experiment-compare-light.png`, `-dark.png` | Every method on the Driving Exam; four optimizers compared |
| `experiment-training-light.png` | A run in progress |

## Layouts and annotated mockups

| File | Shows |
|---|---|
| `layout-split-environment-network-dark.png` | Split layout: Environment and Network side by side, Throttle selected |
| `phone-environment-light.png`, `phone-network-light.png`, `phone-experiment-light.png` | 390 px wide |
| `annotated-environment.png`, `annotated-network.png`, `annotated-experiment-timeline.png`, `annotated-experiment-compare.png` | The design-notes layer: numbered markers on the screen, each explained in the drawer with its category of work |

## `live/`: the real server

Taken by `npm run live-check` while `nd serve` runs, from the run folders on the development
machine (they are not in the repository). Every number is the server's.

| File | Shows |
|---|---|
| `live-environment-keyboard-light.png` | The keyboard driver after a second of the up arrow: rays, telemetry, the Record bar |
| `live-environment-expert-dark.png` | The expert driving at night |
| `live-environment-random8-whole-track-light.png` | Eight random networks, each in its livery (DESIGN_SPEC §7.2) |
| `live-environment-random100-whole-track-light.png` | One hundred random networks, car 43 selected |
| `live-environment-what-if-ghost-light.png` | What-if on the track: the ghost path computed by Python |
| `live-environment-look-back-light.png` | Looking back: Python's `moment` answer, the look-back strip on 1,800 live steps |
| `live-environment-ga-standings-dark.png` | A real GA generation of 20 networks with its ghost, Standings in the drawer |
| `live-network-neuron-dark.png` | The neuron driver's panel (redesign of production's NeuronView), after `set_param` |
| `live-network-random-steer-light.png` | A random network, Steer explained; Flow disabled (needs every layer's contributions) |
| `live-network-what-if-light.png` | What-if answered by Python |
| `live-network-ga-mutation-dark.png` | Mutation mode from `GET .../mutation` |
| `live-experiment-timeline-ga-light.png`, `-dark.png` | A 117-generation GA run at generation 5: the generation panels follow the cursor |
| `live-experiment-timeline-other-light.png` | A random-search run: its metric, checkpoints and exam card |
| `live-experiment-compare-light.png` | The run list grouped by method and sweep; medians from `GET runs/compare` |
| `live-experiment-training-light.png` | Training on the `/ws/runs` feed, with no run in progress |

## `before/`

The current app (commit `13f6b61`), for comparison. Taken with the app's own servers running;
see `UI_AUDIT.md`.
