import { useEffect } from "react";

import { loadRecorded } from "./mock/mockServer";
import { carName, OUTPUT_LABELS, INPUT_LABELS, useApp, type ThemeChoice, type ViewId } from "./state/store";
import { Button, Segmented } from "./ui/controls";
import { Icon, type IconName } from "./ui/Icon";
import { NoteList, NoteMarkers } from "./ui/Notes";
import { EnvironmentView } from "./views/environment/EnvironmentView";
import { Standings } from "./views/environment/Extras";
import { ExperimentView } from "./views/experiment/ExperimentView";
import { NetworkCompact } from "./views/network/NetworkCompact";
import { NetworkView } from "./views/network/NetworkView";

const VIEWS: { id: ViewId; title: string; question: string; icon: IconName }[] = [
  { id: "environment", title: "Environment", question: "What does the car do?", icon: "track" },
  { id: "network", title: "Network", question: "Why did it do that?", icon: "network" },
  { id: "experiment", title: "Experiment", question: "How did it learn to?", icon: "flask" },
];

/** Load the recorded data once, run the playback clock, and keep <html data-theme> in step. */
function useAppLife(): void {
  const setData = useApp((s) => s.setData);
  const theme = useApp((s) => s.theme);

  useEffect(() => {
    loadRecorded().then(setData, (error: Error) => useApp.setState({ loadError: error.message }));
  }, [setData]);

  // The clock: in production the Python server advances the simulation and sends frames; here
  // the recorded frames are stepped through at the same rate (30 steps per second × speed).
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const s = useApp.getState();
      if (!s.playing || s.data === null) return;
      const end = s.data.race.frames.step.length - 1;
      const next = s.t + (dt * 30 * s.speed) / s.data.race.stepsPerFrame;
      if (next >= end) useApp.setState({ t: end, playing: false });
      else useApp.setState({ t: next });
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      document.documentElement.dataset.theme = theme === "auto" ? (media.matches ? "dark" : "light") : theme;
    };
    apply();
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, [theme]);

  // Space plays and pauses, unless the user is typing in a field.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (event.code !== "Space" || (target && /INPUT|SELECT|TEXTAREA|BUTTON/.test(target.tagName))) return;
      event.preventDefault();
      const s = useApp.getState();
      if (s.playing) s.pause();
      else s.play();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

/**
 * The shared selection as a breadcrumb: Run → Generation → Start → Car → Neuron → Moment
 * (PROJECT_CONCEPT §2). It is always on screen, so the user can see what every view is showing
 * and jump to the view where each part is chosen.
 */
function Crumbs() {
  const data = useApp((s) => s.data);
  const car = useApp((s) => s.car);
  const neuron = useApp((s) => s.neuron);
  const connection = useApp((s) => s.connection);
  const viewKey = useApp((s) => s.viewKey);
  const generation = useApp((s) => s.generation);
  if (data === null) return null;
  const info = data.race.scenario.population;
  const layers = data.run.config.network.layer_sizes.length;
  const neuronText =
    neuron !== null
      ? neuron.layer === 0
        ? INPUT_LABELS[neuron.index]
        : neuron.layer === layers - 1
          ? OUTPUT_LABELS[neuron.index]
          : `Hidden ${neuron.layer} · ${neuron.index + 1}`
      : connection !== null
        ? `Weight L${connection.layer + 1} ${connection.from + 1}→${connection.to + 1}`
        : "none";
  const go = (view: ViewId) => () => {
    window.location.hash = `#/${view}`;
  };
  const items: { label: string; value: string; view: ViewId; empty?: boolean; optional?: boolean }[] = [
    { label: "Run", value: info.run.replace(/^\d{8}-\d{6}-/, ""), view: "experiment" },
    { label: "Generation", value: `${generation} of ${info.last_unit}`, view: "experiment" },
    { label: "Start", value: `${info.start + 1} of ${info.n_starts}`, view: "experiment", optional: true },
    { label: "Car", value: carName(data, car), view: "environment" },
    { label: "Neuron", value: neuronText, view: "network", empty: neuron === null && connection === null },
    {
      label: "Moment",
      value: viewKey === null ? "now" : `step ${data.race.frames.step[viewKey]}`,
      view: "environment",
      empty: viewKey === null,
      optional: true,
    },
  ];
  return (
    <nav className="crumbs" aria-label="What is selected" data-testid="crumbs">
      {items.map((item, k) => (
        <span key={item.label} style={{ display: "contents" }}>
          {k > 0 && (
            <span className="crumb-sep" data-optional={item.optional}>
              <Icon name="right" size={12} />
            </span>
          )}
          <button type="button" className="crumb" data-empty={item.empty} data-optional={item.optional} onClick={go(item.view)}>
            <span className="eyebrow">{item.label}</span>
            <span className="crumb-value">{item.value}</span>
          </button>
        </span>
      ))}
    </nav>
  );
}

function TopBar() {
  const view = useApp((s) => s.view);
  const layout = useApp((s) => s.layout);
  const theme = useApp((s) => s.theme);
  const notes = useApp((s) => s.notes);
  const set = useApp((s) => s.set);
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
        <span className="brand-name">
          Neuro<b>Drive</b>
        </span>
      </div>
      <nav className="viewtabs" aria-label="Views">
        {VIEWS.map((v) => (
          <a key={v.id} href={`#/${v.id}`} className="viewtab" title={v.question} aria-current={v.id === view ? "page" : undefined} data-testid={`tab-${v.id}`}>
            <Icon name={v.icon} />
            {v.title}
          </a>
        ))}
      </nav>
      <Crumbs />
      <div className="topbar-right">
        <Button
          variant="hud"
          icon="info"
          pressed={notes}
          onClick={() => set({ notes: !notes, drawerOpen: true, layout: "page" })}
          title="Design notes: what changed and why"
          testId="toggle-notes"
        >
          Notes
        </Button>
        <Segmented
          variant="hud"
          label="Layout"
          value={layout}
          onChange={(id) => set({ layout: id })}
          options={[
            { id: "page", label: "Page", icon: "page", hint: "One view with a companion drawer" },
            { id: "split", label: "Split", icon: "split", hint: "Two views side by side" },
          ]}
        />
        <Segmented
          variant="hud"
          label="Theme"
          value={theme}
          onChange={(id) => set({ theme: id as ThemeChoice })}
          options={[
            { id: "light", label: "", icon: "sun", hint: "Light" },
            { id: "dark", label: "", icon: "moon", hint: "Dark" },
            { id: "auto", label: "", icon: "auto", hint: "Follow the system" },
          ]}
        />
        <span className="conn" data-tip="The prototype replays recorded data; production shows the server connection here">
          <span className="conn-dot" />
          <span className="conn-text">Recorded</span>
        </span>
      </div>
    </header>
  );
}

function MainView({ id }: { id: ViewId }) {
  if (id === "environment") return <EnvironmentView />;
  if (id === "network") return <NetworkView />;
  return <ExperimentView />;
}

/** The companion drawer: the linked view in a compact form, so cause and effect stay together. */
function Drawer({ page }: { page: ViewId }) {
  const open = useApp((s) => s.drawerOpen);
  const tab = useApp((s) => s.drawerTab);
  const notes = useApp((s) => s.notes);
  const set = useApp((s) => s.set);
  const onEnvironment = page === "environment";
  const title = notes ? "Design notes" : onEnvironment ? (tab === "network" ? "Network" : "Standings") : "Environment";
  return (
    <aside className="drawer" data-open={open} data-testid="drawer">
      <div className="drawer-head">
        <Button
          variant="ghost"
          icon={open ? "right" : "left"}
          onClick={() => set({ drawerOpen: !open })}
          title={open ? "Hide the drawer" : `Show ${title}`}
          testId="drawer-toggle"
        />
        {open ? (
          notes ? (
            <span className="eyebrow">Design notes · prototype only</span>
          ) : onEnvironment ? (
            <Segmented
              label="Drawer"
              value={tab}
              onChange={(id) => set({ drawerTab: id })}
              options={[
                { id: "network", label: "Network", icon: "network" },
                { id: "standings", label: "Standings", icon: "chart" },
              ]}
            />
          ) : (
            <span className="eyebrow">Environment · follows the selected car</span>
          )
        ) : (
          <span className="drawer-vertical">{title}</span>
        )}
      </div>
      {open && (
        <div className="drawer-body">
          {notes ? <NoteList /> : onEnvironment ? tab === "network" ? <NetworkCompact /> : <Standings /> : <EnvironmentView compact />}
        </div>
      )}
    </aside>
  );
}

export function App() {
  useAppLife();
  const view = useApp((s) => s.view);
  const layout = useApp((s) => s.layout);
  const error = useApp((s) => s.loadError);
  const partner: ViewId = view === "network" ? "environment" : "network";

  return (
    <div className="app">
      <TopBar />
      <main className="workspace" data-layout={layout}>
        {error !== null ? (
          <div className="view">
            <div className="empty">
              <span className="empty-title">The recorded data could not be loaded</span>
              <span>{error}. Check that public/recorded/ exists (README.md, “Recorded data”).</span>
            </div>
          </div>
        ) : layout === "page" ? (
          <>
            <div className="workspace-main">
              <MainView id={view} />
            </div>
            <Drawer page={view} />
          </>
        ) : (
          <div className="workspace-main">
            <MainView id={view} />
            <MainView id={partner} />
          </div>
        )}
      </main>
      <NoteMarkers />
    </div>
  );
}
