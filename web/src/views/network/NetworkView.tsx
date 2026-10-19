import type { ReactNode } from "react";

import { useLive } from "../../state/live";
import { useSelection } from "../../state/selection";
import { NetworkCompact } from "./NetworkCompact";
import { NetworkDiagram } from "./NetworkDiagram";
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
 * Previous / next car, wrapping around. It changes the shared selection, so the Environment view
 * follows (the selected car's rays, the HUD and the car picker).
 */
function CarStepper({ index, count, compact }: { index: number; count: number; compact: boolean }) {
  const selectCar = useSelection((s) => s.selectCar);
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
      <span className="car-stepper-label" data-testid="car-label">
        Car {index + 1} of {count}
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

  const selected = frame?.selected ?? null;
  const trace = selected?.trace ?? null;
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
    body = <NetworkDiagram network={network} trace={trace} car={selected.index} />;
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
      {body}
    </section>
  );
}
