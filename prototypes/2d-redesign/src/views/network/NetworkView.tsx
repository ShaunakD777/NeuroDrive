import type { Mutation, Recorded } from "../../mock/types";
import { carName, carRole, useApp, useDetail, useRecord, type LineMode } from "../../state/store";
import { Button, LegendItem, Planned, Segmented, Select } from "../../ui/controls";
import { Icon } from "../../ui/Icon";
import { Diagram } from "./Diagram";
import { Inspector } from "./Inspector";
import { NetworkCompact } from "./NetworkCompact";

/** What a recorded car is called in the car picker: its number and why it was recorded. */
function featuredLabel(data: Recorded, i: number): string {
  const info = data.race.scenario.population;
  if (i === info.ghost_car) return `Ghost · generation ${info.ghost_unit} champion`;
  const end = data.race.end.status[i];
  const role = carRole(data, i);
  return `Car ${i + 1} · rank ${info.ranks[i] + 1} · ${role === "child" ? end : role}`;
}

/** Previous / next / pick: changes the shared selection, so the track follows (production: CarStepper). */
function CarPicker() {
  const data = useApp((s) => s.data);
  const car = useApp((s) => s.car);
  const selectCar = useApp((s) => s.selectCar);
  if (data === null) return null;
  const featured = data.race.featured;
  const at = featured.indexOf(car);
  const move = (by: number) => selectCar(featured[(Math.max(0, at) + by + featured.length) % featured.length]);
  const options = featured.map((i) => ({ value: i, label: featuredLabel(data, i) }));
  if (at < 0) options.unshift({ value: car, label: `${carName(data, car)} · not recorded` });
  return (
    <div className="carpicker">
      <Button variant="ghost" icon="left" onClick={() => move(-1)} title="Previous car" testId="car-prev" />
      <Select value={car} options={options} onChange={selectCar} label="Car" />
      <Button variant="ghost" icon="right" onClick={() => move(1)} title="Next car" testId="car-next" />
    </div>
  );
}

/** Where this network came from, in words (production: the lineage banner, task 7.8). */
function originText(data: Recorded, car: number, mutation: Mutation | null): string {
  const info = data.race.scenario.population;
  if (car === info.ghost_car) return `the champion of generation ${info.ghost_unit}, racing as a ghost. It is not part of generation ${info.unit}.`;
  if (mutation === null) return "no lineage recorded for this network.";
  const row = `row ${car} of generation ${info.unit}`;
  const previous = info.unit - 1;
  if (mutation.origin === "elite") return `${row}: an exact copy (elite) of row ${mutation.main_parent} of generation ${previous}. Nothing changed.`;
  if (mutation.origin === "random") return `${row}: a random network of the first generation. It has no parent.`;
  const main = `row ${mutation.main_parent} (rank ${mutation.parents_rank[0] + 1})`;
  const genes = mutation.n_genes - mutation.n_from_other;
  return mutation.crossed
    ? `${row}: a child of ${main} crossed with row ${mutation.other_parent} (rank ${mutation.parents_rank[1] + 1}) of generation ${previous}. ${genes} of its ${mutation.n_genes} numbers come from the first parent, ${mutation.n_from_other} from the second, and ${mutation.n_mutated} were then mutated.`
    : `${row}: a child of ${main} of generation ${previous}, copied and then mutated in ${mutation.n_mutated} of its ${mutation.n_genes} numbers.`;
}

function Legend({ mode, mutation }: { mode: LineMode; mutation: Mutation | null }) {
  const line = (cls: string, width: number) => (
    <svg width="30" height="10" aria-hidden="true">
      <path d="M1 8 C 12 8 18 2 29 2" className={`net-edge ${cls}`} strokeWidth={width} />
    </svg>
  );
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

/**
 * The Network view: the selected car's network, live. Left: the diagram. Right: the inspector
 * rail. In production the diagram came first and the inspector below it, out of sight until
 * scrolled to; side by side, a decision and its explanation are read together.
 */
export function NetworkView() {
  const data = useApp((s) => s.data);
  const car = useApp((s) => s.car);
  const mode = useApp((s) => s.lineMode);
  const viewKey = useApp((s) => s.viewKey);
  const set = useApp((s) => s.set);
  const selectCar = useApp((s) => s.selectCar);
  const lookAt = useApp((s) => s.lookAt);
  const record = useRecord();
  const frame = useDetail();

  const mutation = record?.mutation ?? null;
  const canMutate = mutation !== null && mutation.origin === "child";
  const shownMode: LineMode = mode === "mutation" && !canMutate ? "weights" : mode;
  const sizes = data?.run.config.network.layer_sizes ?? [];
  const parameters = sizes.slice(1).reduce((sum, n, k) => sum + n * sizes[k] + n, 0);

  return (
    <section className="view net-view" data-testid="network-view">
      <header className="view-head">
        <div className="view-title">
          <h2>Network</h2>
          <span className="view-question">Why did it do that?</span>
        </div>
        <CarPicker />
        <div className="net-modes">
          <span className="field-label">Lines show</span>
          <Segmented
            label="Lines show"
            value={shownMode}
            onChange={(id) => set({ lineMode: id })}
            options={[
              { id: "weights", label: "Weights", hint: "What the network has learned: each weight's sign and size" },
              { id: "flow", label: "Flow", hint: "What each connection carries at this moment: weight × value" },
              { id: "mutation", label: "Mutation", disabled: !canMutate, hint: canMutate ? "What changed since the parent" : "Only a child network has a mutation to show" },
              { id: "change", label: "Change", disabled: true, hint: "In a checkpoint replay: what changed since the previous checkpoint" },
            ]}
          />
          {shownMode === "flow" && <Planned>proposed</Planned>}
        </div>
      </header>

      {data === null ? (
        <div className="view-body">
          <div className="skeleton" style={{ height: 320, margin: 16 }} />
        </div>
      ) : record === null || frame === null ? (
        <div className="view-body">
          <div className="empty" data-testid="network-empty">
            <span className="empty-title">{carName(data, car)} was not recorded with its network</span>
            <span>
              In the app every car&apos;s network arrives from Python the moment it is selected. This prototype recorded five of them.
              Pick one:
            </span>
            <div className="empty-actions">
              {data.race.featured.map((i) => (
                <Button key={i} onClick={() => selectCar(i)}>
                  {featuredLabel(data, i)}
                </Button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="net-body">
          <div className="net-main">
            <div className="net-banner" data-testid="lineage-banner">
              <Icon name="dna" />
              <span>
                <b>{carName(data, car)}</b> = {originText(data, car, mutation)}
              </span>
            </div>
            {viewKey !== null && (
              <div className="net-banner net-banner-past" data-testid="past-banner">
                <Icon name="clock" />
                <span>
                  Showing <b>step {data.race.frames.step[viewKey]}</b>, earlier in the drive (now: step{" "}
                  {data.race.frames.step[Math.round(useApp.getState().t)]}).
                </span>
                <Button onClick={() => lookAt(null)}>Back to now</Button>
              </div>
            )}
            <p className="net-intro">
              {sizes.join(" → ")} neurons, {parameters} numbers. The picture is redrawn from Python&apos;s numbers at every step.
            </p>
            <div className="net-wide">
              <Diagram network={record.network} frame={frame} mutation={mutation} mode={shownMode} />
              <Legend mode={shownMode} mutation={mutation} />
            </div>
            {/* Too narrow for 400 curves (a phone): the summary built for a narrow column instead. */}
            <div className="net-narrow">
              <NetworkCompact />
            </div>
          </div>
          <Inspector frame={frame} record={record} />
        </div>
      )}
    </section>
  );
}
