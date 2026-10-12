import type { ConnectionStatus } from "../api/ws";
import { hashFor } from "../nav/route";
import { VIEWS, type ViewId } from "../nav/views";
import type { LayoutMode } from "../state/layout";
import { ConnectionBadge } from "./ConnectionBadge";

const MODES: { id: LayoutMode; label: string; hint: string }[] = [
  { id: "page", label: "Page", hint: "One view at a time, with a companion drawer" },
  { id: "split", label: "Split", hint: "Two views side by side" },
  { id: "all", label: "All", hint: "All three views side by side" },
];

/** The top bar: app name, view tabs, layout switch and connection badge. */
export function TopBar({
  current,
  mode,
  onModeChange,
  status,
}: {
  current: ViewId;
  mode: LayoutMode;
  onModeChange: (mode: LayoutMode) => void;
  status: ConnectionStatus;
}) {
  return (
    <header className="app-header">
      <h1>NeuroDrive</h1>
      {/* Tabs are real links, so back/forward, middle-click and bookmarks all work. */}
      <nav className="tabs" aria-label="Views">
        {VIEWS.map((v) => (
          <a
            key={v.id}
            href={hashFor(v.id)}
            className="tab"
            aria-current={v.id === current ? "page" : undefined}
          >
            {v.title}
          </a>
        ))}
      </nav>
      <div className="header-right">
        <div className="segmented" role="group" aria-label="Layout">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              aria-pressed={m.id === mode}
              title={m.hint}
              onClick={() => onModeChange(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
        <ConnectionBadge status={status} />
      </div>
    </header>
  );
}
