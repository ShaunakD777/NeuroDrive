import type { ReactNode } from "react";

import { useLive } from "../../state/live";
import { NetworkDiagram } from "./NetworkDiagram";
import { NeuronView } from "./NeuronView";

/**
 * The Network page: which picture to show depends on the driver. The random-networks driver gets
 * the multilayer network diagram of the selected car; every other driver keeps the Neuron view
 * (which also explains how to switch to the neuron driver).
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

function MultilayerView({ compact, headerExtra }: { compact: boolean; headerExtra?: ReactNode }) {
  const frame = useLive((s) => s.frame);
  const network = useLive((s) => s.network);
  const connected = useLive((s) => s.status === "connected");

  const selected = frame?.selected ?? null;
  const trace = selected?.trace ?? null;
  // The weights arrive once, right after a car is selected; wait until they are the right car's.
  const ready = selected !== null && trace !== null && network !== null && network.car === selected.index;

  let body: ReactNode;
  if (frame === null) {
    body = <p className="hint">{connected ? "Loading the scenario…" : "Waiting for the server…"}</p>;
  } else if (!ready) {
    body = <p className="hint">Loading the network…</p>;
  } else {
    body = <NetworkDiagram network={network} trace={trace} car={selected.index} compact={compact} />;
  }

  return (
    <section
      className={compact ? "view view-compact network-view" : "view network-view"}
      data-testid={compact ? "network-view-compact" : "network-view"}
    >
      <div className="view-header">
        <h2>{compact ? "Network (compact)" : "Network"}</h2>
        {headerExtra}
      </div>
      {body}
    </section>
  );
}
