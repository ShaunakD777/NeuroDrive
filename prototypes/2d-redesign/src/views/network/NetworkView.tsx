import { useLive } from "@prod/state/live";

import { featuredCars } from "../../source/recorded";
import { carName, carRole, drivesNetworks, hasNetwork, POLICY_LABEL, type Mutation, type Origin, type World } from "../../source/model";
import { useWorld } from "../../source/world";
import { shortRun, useApp, type LineMode } from "../../state/store";
import { Button, LegendItem, Planned, Segmented, Select } from "../../ui/controls";
import { Icon } from "../../ui/Icon";
import { SourceWaiting } from "../../ui/SourceWaiting";
import { Diagram } from "./Diagram";
import { Inspector } from "./Inspector";
import { NetworkCompact } from "./NetworkCompact";
import { NeuronPanel } from "./NeuronPanel";

/** How a car is listed in the picker: its number and, in a GA race, its rank and role. */
function carLabel(world: World, i: number): string {
  const info = world.scenario.population;
  if (info === null) return carName(world.scenario, i);
  if (i === info.ghost_car) return `Ghost · generation ${info.ghost_unit} champion`;
  const role = carRole(world.scenario, i);
  return `Car ${i + 1} · rank ${info.ranks[i] + 1}${role === "child" ? "" : ` · ${role}`}`;
}

/** Previous / next / pick: changes the shared selection, so the track follows (production: CarStepper). */
function CarPicker({ world }: { world: World }) {
  const data = useApp((s) => s.data);
  const selectCar = useApp((s) => s.selectCar);
  const car = world.selected;
  // The recording has five networks; live, every car's network arrives when it is selected.
  const choices = world.source === "recorded" ? (featuredCars(data) ?? []) : world.cars.map((_, i) => i);
  if (choices.length <= 1 && world.cars.length <= 1) return null;
  const at = choices.indexOf(car);
  const move = (by: number) => selectCar(choices[(Math.max(0, at) + by + choices.length) % choices.length]);
  const options = choices.map((i) => ({ value: i, label: carLabel(world, i) }));
  if (at < 0) options.unshift({ value: car, label: `${carLabel(world, car)} · not recorded` });
  return (
    <div className="carpicker">
      <Button variant="ghost" icon="left" onClick={() => move(-1)} title="Previous car" testId="car-prev" />
      <Select value={car} options={options} onChange={selectCar} label="Car" testId="car-pick" />
      <Button variant="ghost" icon="right" onClick={() => move(1)} title="Next car" testId="car-next" />
    </div>
  );
}

/** Where this network came from, in words (production: the lineage and replay banners, tasks 5.9 and 7.8). */
function originText(world: World, origin: Origin): string | null {
  const { scenario } = world;
  const car = world.selected;
  const info = scenario.population;
  if (scenario.checkpoint !== null) {
    const ck = scenario.checkpoint;
    const base = `checkpoint ${ck.unit} of ${shortRun(ck.run)}, driving the exam's starts`;
    return origin.diff !== null ? `${base}. "Change" shows what learning changed since checkpoint ${origin.diffFrom}.` : `${base}${origin.note ? ` (${origin.note})` : ""}.`;
  }
  if (info === null) return scenario.policy === "random_networks" ? "a random network: weights drawn from the seed, never trained." : null;
  if (origin.note !== null) return `${origin.note}.`;
  const mutation: Mutation | null = origin.mutation;
  if (mutation === null) return "looking up its parents…";
  const row = `row ${car} of generation ${info.unit}`;
  const previous = info.unit - 1;
  if (mutation.origin === "elite") return `${row}: an exact copy (elite) of row ${mutation.main_parent} of generation ${previous}. Nothing changed.`;
  const main = `row ${mutation.main_parent} (rank ${mutation.parents_rank[0] + 1})`;
  const genes = mutation.n_genes - mutation.n_from_other;
  return mutation.crossed
    ? `${row}: a child of ${main} crossed with row ${mutation.other_parent} (rank ${mutation.parents_rank[1] + 1}) of generation ${previous}. ${genes} of its ${mutation.n_genes} numbers come from the first parent, ${mutation.n_from_other} from the second, and ${mutation.n_mutated} were then mutated.`
    : `${row}: a child of ${main} of generation ${previous}, copied and then mutated in ${mutation.n_mutated} of its ${mutation.n_genes} numbers.`;
}

function Legend({ mode, origin }: { mode: LineMode; origin: Origin }) {
  const line = (cls: string, width: number) => (
    <svg width="30" height="10" aria-hidden="true">
      <path d="M1 8 C 12 8 18 2 29 2" className={`net-edge ${cls}`} strokeWidth={width} />
    </svg>
  );
  const mutation = origin.mutation;
  return (
    <div className="legend net-legend" data-testid="network-legend">
      {mode === "weights" && (
        <>
          <LegendItem swatch={line("net-edge-pos", 2.5)}>positive weight</LegendItem>
          <LegendItem swatch={line("net-edge-neg", 2.5)}>negative weight</LegendItem>
          <LegendItem swatch={line("net-edge-pos", 0.8)}>thin = small |weight|, thick = large</LegendItem>
        </>
      )}
      {mode === "flow" && (
        <>
          <LegendItem swatch={line("net-edge-pos", 2.5)}>carries a positive amount now (weight × value)</LegendItem>
          <LegendItem swatch={line("net-edge-neg", 2.5)}>a negative amount</LegendItem>
          <LegendItem swatch={line("net-edge-pos", 0.8)}>thicker = more, within its layer</LegendItem>
        </>
      )}
      {mode === "change" && origin.diff !== null && (
        <>
          <LegendItem swatch={line("net-edge-pos", 2.5)}>weight rose since checkpoint {origin.diffFrom}</LegendItem>
          <LegendItem swatch={line("net-edge-neg", 2.5)}>weight fell</LegendItem>
          <span>thicker = bigger change (largest {origin.diff.max_abs_delta.toFixed(4)}); unchanged weights are not drawn</span>
        </>
      )}
      {mode === "mutation" && mutation !== null && (
        <>
          <LegendItem swatch={line("net-edge-pos", 2.5)}>mutation raised the weight</LegendItem>
          <LegendItem swatch={line("net-edge-neg", 2.5)}>lowered it</LegendItem>
          {mutation.n_from_other > 0 && <LegendItem swatch={line("net-edge-other", 1)}>from the second parent, row {mutation.other_parent}</LegendItem>}
          <LegendItem swatch={<span className="legend-mutated" aria-hidden="true" />}>mutated bias</LegendItem>
          <span>thicker = bigger nudge (largest {mutation.max_abs_delta.toFixed(3)}); unchanged weights fade out</span>
        </>
      )}
      <LegendItem
        swatch={
          <svg width="46" height="14" aria-hidden="true">
            <circle cx="7" cy="7" r="6" className="net-node-base" />
            <circle cx="7" cy="7" r="6" className="net-fill-pos" fillOpacity="0.25" />
            <circle cx="7" cy="7" r="6" className="net-node-ring" />
            <circle cx="23" cy="7" r="6" className="net-fill-pos" />
            <circle cx="39" cy="7" r="6" className="net-fill-neg" />
          </svg>
        }
      >
        circle = the neuron&apos;s value now: stronger colour = larger; blue positive, orange negative
      </LegendItem>
    </div>
  );
}

/** The selected car has no network to draw: say why, and (live) offer drivers that have one. */
function NoNetwork({ world }: { world: World }) {
  const data = useApp((s) => s.data);
  const selectCar = useApp((s) => s.selectCar);
  const loadScenario = useLive((s) => s.loadScenario);
  if (world.source === "recorded") {
    const featured = featuredCars(data) ?? [];
    return (
      <div className="empty" data-testid="network-empty">
        <span className="empty-title">{carName(world.scenario, world.selected)} was not recorded with its network</span>
        <span>Live, every car&apos;s network arrives from Python the moment it is selected. The recording kept five of them. Pick one:</span>
        <div className="empty-actions">
          {featured.map((i) => (
            <Button key={i} onClick={() => selectCar(i)}>
              {carLabel(world, i)}
            </Button>
          ))}
        </div>
      </div>
    );
  }
  if (drivesNetworks(world.scenario.policy)) return <SourceWaiting what="the selected car's network" />;
  return (
    <div className="empty" data-testid="network-empty">
      <span className="empty-title">{POLICY_LABEL[world.scenario.policy].replace(/ \(.*\)$/, "")} has no network</span>
      <span>This driver decides without a neural network, so there is nothing to draw here. Choose a driver that uses one:</span>
      <div className="empty-actions">
        <Button icon="network" onClick={() => loadScenario({ policy: "random_networks", n_cars: Math.max(world.scenario.n_cars, 10) })}>
          Random networks
        </Button>
        <Button icon="network" onClick={() => loadScenario({ policy: "neuron" })}>
          One neuron
        </Button>
      </div>
      <span className="muted">Or race a GA generation, or replay a checkpoint from Experiment → Timeline.</span>
    </div>
  );
}

/**
 * The Network view: the selected car's network, live. Left: the diagram. Right: the inspector
 * rail. In production the diagram came first and the inspector below it, out of sight until
 * scrolled to; side by side, a decision and its explanation are read together.
 */
export function NetworkView() {
  const world = useWorld();
  const mode = useApp((s) => s.lineMode);
  const set = useApp((s) => s.set);
  const lookAt = useApp((s) => s.lookAt);

  const origin = world?.origin ?? null;
  const canMutate = origin?.mutation?.origin === "child";
  const canChange = origin?.diff != null;
  const canFlow = world?.flow != null;
  const available: Record<LineMode, boolean> = { weights: true, flow: canFlow, mutation: canMutate, change: canChange };
  const shownMode: LineMode = available[mode] ? mode : "weights";
  const network = world?.network ?? null;
  const shown = world?.shown ?? null;
  const sizes = network?.layer_sizes ?? [];
  const parameters = sizes.slice(1).reduce((sum, n, k) => sum + n * sizes[k] + n, 0);
  const text = world !== null && origin !== null ? originText(world, origin) : null;

  return (
    <section className="view net-view" data-testid="network-view">
      <header className="view-head">
        <div className="view-title">
          <h2>Network</h2>
          <span className="view-question">Why did it do that?</span>
        </div>
        {world !== null && <CarPicker world={world} />}
        {network !== null && (
          <div className="net-modes">
            <span className="field-label">Lines show</span>
            <Segmented
              label="Lines show"
              value={shownMode}
              onChange={(id) => set({ lineMode: id })}
              options={[
                { id: "weights", label: "Weights", hint: "What the network has learned: each weight's sign and size" },
                {
                  id: "flow",
                  label: "Flow",
                  disabled: !canFlow,
                  hint: canFlow
                    ? "What each connection carries at this moment: weight × value"
                    : "Needs every layer's weight × value from Python, which it does not send yet (PRODUCTION_MAPPING.md §6.1)",
                },
                { id: "mutation", label: "Mutation", disabled: !canMutate, hint: canMutate ? "What changed since the parent" : "Only a child of a GA generation has a mutation to show" },
                { id: "change", label: "Change", disabled: !canChange, hint: canChange ? "What learning changed since the previous checkpoint" : "In a checkpoint replay: what changed since the previous checkpoint" },
              ]}
            />
            {shownMode === "flow" && <Planned>proposed</Planned>}
          </div>
        )}
      </header>

      {world === null ? (
        <div className="view-body">
          <SourceWaiting what="a car to explain" height={320} />
        </div>
      ) : shown?.neuron ? (
        <NeuronPanel neuron={shown.neuron} rays={shown.rays} />
      ) : network === null || !hasNetwork(shown) ? (
        <div className="view-body">
          <NoNetwork world={world} />
        </div>
      ) : (
        <div className="net-body">
          <div className="net-main">
            {text !== null && (
              <div className="net-banner" data-testid={world.scenario.checkpoint ? "replay-banner" : "lineage-banner"}>
                <Icon name={world.scenario.checkpoint ? "clock" : "dna"} />
                <span>
                  <b>{carName(world.scenario, world.selected)}</b> = {text}
                </span>
              </div>
            )}
            {world.lookStep !== null && (
              <div className="net-banner net-banner-past" data-testid="past-banner">
                <Icon name="clock" />
                <span>
                  Showing <b>step {world.lookStep}</b>, earlier in the drive (now: step {world.step}).
                </span>
                <Button onClick={() => lookAt(null)} testId="past-live">
                  Back to now
                </Button>
              </div>
            )}
            <p className="net-intro">
              {sizes.join(" → ")} neurons, {parameters} numbers. The picture is redrawn from Python&apos;s numbers at every step.
            </p>
            <div className="net-wide">
              <Diagram network={network} frame={shown} mutation={origin?.mutation ?? null} diff={origin?.diff ?? null} flow={world.flow} mode={shownMode} />
              <Legend mode={shownMode} origin={origin ?? { mutation: null, diff: null, diffFrom: null, note: null }} />
            </div>
            {/* Too narrow for 400 curves (a phone): the summary built for a narrow column instead. */}
            <div className="net-narrow">
              <NetworkCompact />
            </div>
          </div>
          <Inspector frame={shown} network={network} />
        </div>
      )}
    </section>
  );
}
