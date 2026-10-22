import type { ReactNode } from "react";

import type { Explanation } from "../../api/generated";

import { useLive } from "../../state/live";
import { useMoment, usePlayback } from "../../state/playback";
import { useSelection } from "../../state/selection";
import { NetworkCompact } from "./NetworkCompact";
import { NetworkDiagram } from "./NetworkDiagram";
import { OutputInspector } from "./OutputInspector";
import { WhatIfPanel } from "./WhatIfPanel";
import { NeuronView } from "./NeuronView";

/**
 * The Network page: which picture to show depends on the driver. The random-networks driver gets
 * the selected car's multilayer network (the full diagram, or the compact summary in a drawer);
 * every other driver keeps the Neuron view (which also explains how to switch to it).
 */
export function NetworkView({
  compact = false,
  headerExtra,
}: {
  compact?: boolean;
  headerExtra?: ReactNode;
}) {
  const policy = useLive((s) => s.scenario?.policy ?? null);
  if (policy !== "random_networks") return <NeuronView compact={compact} headerExtra={headerExtra} />;
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

function MultilayerView({ compact, headerExtra }: { compact: boolean; headerExtra?: ReactNode }) {
  const frame = useLive((s) => s.frame);
  const network = useLive((s) => s.network);
  const connected = useLive((s) => s.status === "connected");

  // Looking at an earlier step (task 2.11): the network as it was then.
  const moment = useMoment();
  const backToLive = usePlayback((s) => s.backToLive);
  const selected = frame?.selected ?? null;
  const trace = moment?.trace ?? selected?.trace ?? null;
  const count = frame?.cars.x.length ?? 0;
  // The weights arrive once, right after a car is selected; wait until they are the right car's.
  const ready = selected !== null && trace !== null && network !== null && network.car === selected.index;

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
        <NetworkDiagram network={network} trace={trace} car={selected.index} atStep={moment?.step ?? null} />
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
