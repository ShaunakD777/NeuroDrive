# Screenshots

Every image except those in `before/` is the running prototype, taken by
`npm run screenshots` (Chromium through Playwright). Desktop images are 1440 × 900 unless the
content is taller; phone images are 390 × 844.

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

## `before/`

The current app (commit `13f6b61`), for comparison. Taken with the app's own servers running;
see `UI_AUDIT.md`.
