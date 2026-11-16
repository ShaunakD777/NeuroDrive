import { useState, type ReactNode } from "react";

import type { Explanation } from "../../api/generated";

import { useLive } from "../../state/live";
import { useMoment, usePlayback } from "../../state/playback";
import { useSelection } from "../../state/selection";
import { describeStarts } from "../environment/status";
import { NetworkCompact } from "./NetworkCompact";
import { NetworkDiagram } from "./NetworkDiagram";
import { OutputInspector } from "./OutputInspector";
import { WhatIfPanel } from "./WhatIfPanel";
import { NeuronView } from "./NeuronView";
import { originText } from "./diff";
import { useCheckpointChange, useMutation } from "./hooks";

/**
 * The Network page: which picture to show depends on the driver. The random-networks driver, a
 * replayed checkpoint (task 5.9) and a raced GA generation (task 7.6) get the selected car's
 * multilayer network (the full diagram, or
 * the compact summary in a drawer); every other driver keeps the Neuron view (which also explains
 * how to switch to it).
 */
export function NetworkView({
  compact = false,
  headerExtra,
}: {
  compact?: boolean;
  headerExtra?: ReactNode;
}) {
  const policy = useLive((s) => s.scenario?.policy ?? null);
  // A raced GA generation (task 7.6) is one network per car, like the random networks.
  if (policy !== "random_networks" && policy !== "checkpoint" && policy !== "population")
    return <NeuronView compact={compact} headerExtra={headerExtra} />;
  return <MultilayerView compact={compact} headerExtra={headerExtra} />;
}

/**
 * Choose the car from the Network page: previous / next (wrapping round) or straight to any car
 * from the drop-down. It changes the shared selection, so the Environment view follows (the
 * selected car's rays, the HUD and its own car picker).
 */
function CarStepper({ index: shown, count, compact }: { index: number; count: number; compact: boolean }) {
  const selectCar = useSelection((s) => s.selectCar);
  // Count from the car last asked for, not the one the server last confirmed, so two quick
  // clicks on › move two cars even before the first answer arrives.
  const index = useSelection((s) => s.carId) ?? shown;
  if (count < 2) return <span className="car-stepper-label">Car {index + 1}</span>;
  return (
    <span className={compact ? "car-stepper car-stepper-compact" : "car-stepper"}>
      <button
        type="button"
        data-testid="car-prev"
        aria-label="Previous car"
        onClick={() => selectCar((index - 1 + count) % count)}
      >
        ‹
      </button>
      <select
        data-testid="car-pick"
        aria-label="Car"
        value={index}
        onChange={(e) => {
          selectCar(Number(e.target.value));
          e.target.blur(); // so the arrow keys drive instead of changing this menu
        }}
      >
        {Array.from({ length: count }, (_, i) => (
          <option key={i} value={i}>
            Car {i + 1}
          </option>
        ))}
      </select>
      <span className="car-stepper-label" data-testid="car-label">
        of {count}
      </span>
      <button
        type="button"
        data-testid="car-next"
        aria-label="Next car"
        onClick={() => selectCar((index + 1) % count)}
      >
        ›
      </button>
    </span>
  );
}

/** The full view's "show change" choice, kept while scrubbing from checkpoint to checkpoint. */
let showChangeChoice = true;

/** The "show mutation" choice in population mode (task 7.8), kept from car to car. */
let showMutationChoice = false;

function MultilayerView({ compact, headerExtra }: { compact: boolean; headerExtra?: ReactNode }) {
  const frame = useLive((s) => s.frame);
  const checkpoint = useLive((s) => s.scenario?.checkpoint ?? null);
  const startShift = useLive((s) => s.scenario?.start_shift ?? null);
  const change = useCheckpointChange(compact ? null : checkpoint);
  const [showChange, setShowChange] = useState(showChangeChoice);
  const network = useLive((s) => s.network);
  const connected = useLive((s) => s.status === "connected");
  const population = useLive((s) => s.scenario?.population ?? null);
  const [showMutation, setShowMutation] = useState(showMutationChoice);

  // Looking at an earlier step (task 2.11): the network as it was then.
  const moment = useMoment();
  const backToLive = usePlayback((s) => s.backToLive);
  const selected = frame?.selected ?? null;
  const trace = moment?.trace ?? selected?.trace ?? null;
  const count = frame?.cars.x.length ?? 0;
  // The weights arrive once, right after a car is selected; wait until they are the right car's.
  const ready = selected !== null && trace !== null && network !== null && network.car === selected.index;
  const origin = useMutation(compact ? null : population, selected?.index ?? null);
  const mutation = showMutation ? origin.mutation : null;

  let body: ReactNode;
  if (frame === null) {
    body = <p className="hint">{connected ? "Loading the scenario…" : "Waiting for the server…"}</p>;
  } else if (!ready) {
    body = <p className="hint">Loading the network…</p>;
  } else if (compact) {
    body = <NetworkCompact network={network} trace={trace} />;
  } else {
    body = (
      <>
        <NetworkDiagram
          network={network}
          trace={trace}
          car={selected.index}
          atStep={moment?.step ?? null}
          change={showChange ? change.diff : null}
          mutation={mutation}
        />
        <Inspection explain={moment === null ? (selected.explain ?? null) : moment.explain} />
      </>
    );
  }

  return (
    <section
      className={compact ? "view view-compact network-view" : "view network-view"}
      data-testid={compact ? "network-view-compact" : "network-view"}
    >
      <div className="view-header">
        <h2>{compact ? "Network (compact)" : "Network"}</h2>
        {selected !== null && <CarStepper index={selected.index} count={count} compact={compact} />}
        {headerExtra}
      </div>
      {checkpoint !== null && !compact && (
        <div className="replay-banner" data-testid="replay-banner" data-run={checkpoint.run} data-unit={checkpoint.unit}>
          <span>
            Replaying checkpoint <strong>unit {checkpoint.unit}</strong> of {checkpoint.run}: the same network
            drives every car, from {describeStarts(startShift)}.
          </span>
          <label className="replay-toggle">
            <input
              type="checkbox"
              data-testid="show-change"
              checked={showChange && change.from !== null}
              disabled={change.from === null}
              onChange={(e) => {
                showChangeChoice = e.target.checked;
                setShowChange(e.target.checked);
              }}
            />
            {change.from === null
              ? " Colour lines by change: this is the first checkpoint, nothing to compare"
              : ` Colour lines by change since unit ${change.from}`}
          </label>
        </div>
      )}
      {population !== null && !compact && selected !== null && (
        <div
          className="replay-banner"
          data-testid="lineage-banner"
          data-unit={population.unit}
          data-row={selected.index}
          data-origin={origin.mutation?.origin ?? "none"}
        >
          <span>
            Car {selected.index + 1}
            {selected.index < population.size ? ` = row ${selected.index} of generation ${population.unit}` : ""}:{" "}
            {origin.mutation !== null ? originText(origin.mutation) : (origin.note ?? `no lineage: ${origin.error}`)}.
          </span>
          {origin.mutation !== null && origin.mutation.origin === "child" && (
            <label className="replay-toggle">
              <input
                type="checkbox"
                data-testid="show-mutation"
                checked={showMutation}
                onChange={(e) => {
                  showMutationChoice = e.target.checked;
                  setShowMutation(e.target.checked);
                }}
              />{" "}
              Colour lines by mutation since the parent
            </label>
          )}
        </div>
      )}
      {moment !== null && frame !== null && (
        <div className="past-banner" data-testid="past-banner">
          <span>
            Showing step <strong>{moment.step}</strong> of car {moment.car + 1}, earlier in the drive
            (now: step {frame.step}).
          </span>
          <button type="button" data-testid="past-live" onClick={() => backToLive()}>
            Back to now
          </button>
        </div>
      )}
      {body}
    </section>
  );
}

/**
 * The output inspector and the what-if panel, once an output neuron is selected (task 2.10).
 * `explain` is the present one, or an earlier step's while one is looked at (task 2.11).
 */
function Inspection({ explain }: { explain: Explanation | null }) {
  const output = useSelection((s) => s.output);
  const network = useLive((s) => s.network);
  if (output === null) {
    return (
      <p className="hint inspection-hint" data-testid="inspection-hint">
        Click Steer or Throttle (on the right of the diagram) to see why it has its value and what it
        reacts to.
      </p>
    );
  }
  if (explain === null || network === null) return null;
  const index = output === "steer" ? 0 : 1;
  return (
    <div className="inspection">
      <OutputInspector explain={explain} network={network} output={index} />
      <WhatIfPanel />
    </div>
  );
}
