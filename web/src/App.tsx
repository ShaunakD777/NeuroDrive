import { useEffect, useState } from "react";
import { Group, Panel, Separator } from "react-resizable-panels";

import { connectLive, type ConnectionStatus } from "./api/ws";
import { ConnectionBadge } from "./components/ConnectionBadge";
import { ViewPanel } from "./components/ViewPanel";

export function App() {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");

  useEffect(() => connectLive(setStatus), []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>NeuroDrive</h1>
        <ConnectionBadge status={status} />
      </header>
      <Group orientation="horizontal" className="panels">
        <Panel id="panel-environment" defaultSize="34" minSize="15">
          <ViewPanel title="Environment" comingIn="task 1.8" />
        </Panel>
        <Separator className="separator" />
        <Panel id="panel-network" defaultSize="33" minSize="15">
          <ViewPanel title="Neural Network" comingIn="task 1.9 and Phase 2" />
        </Panel>
        <Separator className="separator" />
        <Panel id="panel-experiment" defaultSize="33" minSize="15">
          <ViewPanel title="Experiment" comingIn="Phase 4 and later" />
        </Panel>
      </Group>
    </div>
  );
}
