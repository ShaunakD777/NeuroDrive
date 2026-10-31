import type { ReactNode } from "react";
import { Group, Panel, Separator } from "react-resizable-panels";

import { useLiveConnection } from "./api/useLiveConnection";
import { CompanionDrawer } from "./components/CompanionDrawer";
import { TopBar } from "./components/TopBar";
import { ViewPanel } from "./components/ViewPanel";
import { useRoute } from "./nav/route";
import { VIEWS, viewInfo, type ViewId } from "./nav/views";
import { effectivePartner, useLayout } from "./state/layout";
import { useLive } from "./state/live";
import { EnvironmentView } from "./views/environment/EnvironmentView";
import { ExperimentView } from "./views/experiment/ExperimentView";
import { NetworkView } from "./views/network/NetworkView";

export function App() {
  // The connection belongs to the app, not to a page, so it survives page changes.
  useLiveConnection();
  const status = useLive((s) => s.status);

  const current = useRoute();
  const { mode, setMode } = useLayout();

  return (
    <div className="app">
      <TopBar current={current} mode={mode} onModeChange={setMode} status={status} />
      <main className="workspace">
        {mode === "page" && <PageLayout current={current} />}
        {mode === "split" && <SplitLayout current={current} />}
        {mode === "all" && <AllLayout current={current} />}
      </main>
    </div>
  );
}

function MainView({ id, headerExtra }: { id: ViewId; headerExtra?: ReactNode }) {
  if (id === "environment") return <EnvironmentView headerExtra={headerExtra} />;
  if (id === "network") return <NetworkView headerExtra={headerExtra} />;
  if (id === "experiment") return <ExperimentView headerExtra={headerExtra} />;
  const v = viewInfo(id);
  return <ViewPanel title={v.title} comingIn={v.comingIn} headerExtra={headerExtra} />;
}

/** One view, full size, with its companion drawer. */
function PageLayout({ current }: { current: ViewId }) {
  const open = useLayout((s) => s.drawerOpen[current]);
  const toggleDrawer = useLayout((s) => s.toggleDrawer);
  return (
    <div className="page" data-testid={`page-${current}`}>
      <div className="page-main">
        <MainView id={current} />
      </div>
      <CompanionDrawer page={current} open={open} onToggle={() => toggleDrawer(current)} />
    </div>
  );
}

/** The current view on the left, a chosen partner on the right. */
function SplitLayout({ current }: { current: ViewId }) {
  const splitPartner = useLayout((s) => s.splitPartner);
  const setSplitPartner = useLayout((s) => s.setSplitPartner);
  const partner = effectivePartner(current, splitPartner);

  const picker = (
    <label className="picker">
      <span className="visually-hidden">Second view</span>
      <select value={partner} onChange={(e) => setSplitPartner(e.target.value as ViewId)}>
        {VIEWS.filter((v) => v.id !== current).map((v) => (
          <option key={v.id} value={v.id}>
            {v.title}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <Group orientation="horizontal" className="panels">
      <Panel id={`split-${current}`} defaultSize="50" minSize="20">
        <MainView id={current} />
      </Panel>
      <Separator className="separator" />
      <Panel id={`split-${partner}`} defaultSize="50" minSize="20">
        <MainView id={partner} headerExtra={picker} />
      </Panel>
    </Group>
  );
}

/** All three views side by side; the current one is outlined. */
function AllLayout({ current }: { current: ViewId }) {
  return (
    <Group orientation="horizontal" className="panels">
      {VIEWS.map((v, i) => (
        <AllPanel key={v.id} id={v.id} isCurrent={v.id === current} withSeparator={i > 0} />
      ))}
    </Group>
  );
}

function AllPanel({
  id,
  isCurrent,
  withSeparator,
}: {
  id: ViewId;
  isCurrent: boolean;
  withSeparator: boolean;
}) {
  return (
    <>
      {withSeparator && <Separator className="separator" />}
      <Panel
        id={`panel-${id}`}
        defaultSize={id === "environment" ? "34" : "33"}
        minSize="15"
        className={isCurrent ? "panel-current" : undefined}
      >
        <MainView id={id} />
      </Panel>
    </>
  );
}
