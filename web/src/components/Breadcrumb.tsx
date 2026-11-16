import type { ScenarioMessage } from "../api/generated";
import { hashFor } from "../nav/route";
import type { ViewId } from "../nav/views";
import { useLive } from "../state/live";
import { usePlayback } from "../state/playback";
import { type ConnectionRef, type NeuronRef, useSelection } from "../state/selection";
import { nodeLabel } from "../views/network/networkLayout";
import { Icon } from "./ui/Icon";

// The shared selection as a breadcrumb in the top bar (task S1, DESIGN_SPEC.md §5.1): where the
// cars' behaviour comes from (a run's generation or checkpoint, or a driver), then the car, the
// neuron and the moment, in the order of PROJECT_CONCEPT.md §2. It only reads the stores; each part
// is a button that opens the view where that part is chosen.

export interface Crumb {
  /** The name of the part, shown as a small capital label ("Run", "Car"). */
  label: string;
  value: string;
  /** The view where this part is chosen. */
  view: ViewId;
  /** Nothing chosen yet: shown muted ("none", "now"). */
  empty?: boolean;
  /** Dropped first when the top bar is narrow. */
  optional?: boolean;
}

export interface CrumbInput {
  scenario: ScenarioMessage | null;
  /** Index of the selected car (the server's `frame.selected.index`), null before the first frame. */
  selected: number | null;
  /** Number of layers of the selected car's network including inputs and outputs; 0 if none. */
  layers: number;
  neuron: NeuronRef | null;
  connection: ConnectionRef | null;
  /** The look-back step, or null for "now". */
  viewStep: number | null;
}

const DRIVER_LABEL: Record<ScenarioMessage["policy"], string> = {
  keyboard: "Keyboard",
  constant: "Constant",
  neuron: "Neuron",
  random_networks: "Random networks",
  expert: "Expert",
  checkpoint: "Checkpoint replay",
  population: "GA generation",
};

/** A run id without its date and time prefix: `20261110-101500-ga-default` → `ga-default`. */
export function shortRun(runId: string): string {
  return runId.replace(/^\d{8}-\d{6}-/, "");
}

function originCrumbs(scenario: ScenarioMessage): Crumb[] {
  const info = scenario.population;
  if (info !== null)
    return [
      { label: "Run", value: shortRun(info.run), view: "experiment" },
      { label: "Generation", value: `${info.unit} of ${info.last_unit}`, view: "experiment" },
      {
        label: "Start",
        value: `${info.start + 1} of ${info.n_starts}`,
        view: "experiment",
        optional: true,
      },
    ];
  if (scenario.checkpoint !== null)
    return [
      { label: "Run", value: shortRun(scenario.checkpoint.run), view: "experiment" },
      { label: "Checkpoint", value: `unit ${scenario.checkpoint.unit}`, view: "experiment" },
      { label: "Track", value: scenario.track, view: "environment", optional: true },
    ];
  return [
    { label: "Driver", value: DRIVER_LABEL[scenario.policy], view: "environment" },
    { label: "Track", value: scenario.track, view: "environment", optional: true },
  ];
}

function carValue(scenario: ScenarioMessage, selected: number | null): string {
  if (selected === null) return "none";
  const info = scenario.population;
  if (info !== null && info.ghost_car === selected) return `Ghost · gen ${info.ghost_unit} champion`;
  return `Car ${selected + 1}`;
}

function neuronValue({ layers, neuron, connection }: CrumbInput): string {
  if (neuron !== null) {
    if (layers === 0) return `Layer ${neuron.layer} · ${neuron.index + 1}`;
    // "Hidden 1, neuron 6" reads "Hidden 1 · 6" in the narrow crumb.
    return nodeLabel(neuron.layer, neuron.index, layers).replace(/, neuron /, " · ");
  }
  if (connection !== null)
    return `Weight L${connection.layer + 1} ${connection.from + 1}→${connection.to + 1}`;
  return "none";
}

/** The breadcrumb's parts for the current selection; empty before the first scenario arrives. */
export function crumbsFor(input: CrumbInput): Crumb[] {
  const { scenario, neuron, connection, viewStep } = input;
  if (scenario === null) return [];
  return [
    ...originCrumbs(scenario),
    {
      label: "Car",
      value: carValue(scenario, input.selected),
      view: "environment",
      empty: input.selected === null,
    },
    {
      label: "Neuron",
      value: neuronValue(input),
      view: "network",
      empty: neuron === null && connection === null,
    },
    {
      label: "Moment",
      value: viewStep === null ? "now" : `step ${viewStep}`,
      view: "environment",
      empty: viewStep === null,
      optional: true,
    },
  ];
}

export function Breadcrumb() {
  const scenario = useLive((s) => s.scenario);
  const selected = useLive((s) => s.frame?.selected?.index ?? null);
  const layers = useLive((s) => s.network?.layer_sizes.length ?? 0);
  const neuron = useSelection((s) => s.neuron);
  const connection = useSelection((s) => s.connection);
  const viewStep = usePlayback((s) => s.viewStep);
  const items = crumbsFor({ scenario, selected, layers, neuron, connection, viewStep });
  if (items.length === 0) return null;
  return (
    <nav className="crumbs" aria-label="What is selected" data-testid="breadcrumb">
      {items.map((item, k) => (
        <span key={item.label} className="crumb-item" data-optional={item.optional}>
          {k > 0 && (
            <span className="crumb-sep" aria-hidden="true">
              <Icon name="right" size={12} />
            </span>
          )}
          <button
            type="button"
            className="crumb"
            data-empty={item.empty}
            data-testid={`crumb-${item.label.toLowerCase()}`}
            title={`Chosen in the ${item.view[0].toUpperCase()}${item.view.slice(1)} view`}
            onClick={() => {
              window.location.hash = hashFor(item.view);
            }}
          >
            <span className="eyebrow">{item.label}</span>
            <span className="crumb-value">{item.value}</span>
          </button>
        </span>
      ))}
    </nav>
  );
}
