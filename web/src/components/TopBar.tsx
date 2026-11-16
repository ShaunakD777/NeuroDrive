import type { ConnectionStatus } from "../api/ws";
import { hashFor } from "../nav/route";
import { VIEWS, type ViewId } from "../nav/views";
import { type LayoutMode, type ThemeChoice, useLayout } from "../state/layout";
import { Breadcrumb } from "./Breadcrumb";
import { ConnectionBadge } from "./ConnectionBadge";
import { Icon } from "./ui/Icon";
import { Segmented } from "./ui/controls";

/**
 * The top bar (task S1, DESIGN_SPEC.md §5.1): dark in both themes, like an instrument's bezel.
 * Brand mark, view tabs, the selection breadcrumb, then the layout switch, the theme switch and
 * the connection badge.
 */
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
  const theme = useLayout((s) => s.theme);
  const setTheme = useLayout((s) => s.setTheme);
  return (
    <header className="topbar">
      <h1 className="brand">
        {/* Three slanted kerb blocks: the only place the kerb motif appears outside the track. */}
        <span className="brand-mark" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
        <span className="brand-name">
          Neuro<b>Drive</b>
        </span>
      </h1>
      {/* Tabs are real links, so back/forward, middle-click and bookmarks all work. */}
      <nav className="viewtabs" aria-label="Views">
        {VIEWS.map((v) => (
          <a
            key={v.id}
            href={hashFor(v.id)}
            className="viewtab"
            title={v.question}
            aria-current={v.id === current ? "page" : undefined}
          >
            <Icon name={v.icon} />
            {v.title}
          </a>
        ))}
      </nav>
      <Breadcrumb />
      <div className="topbar-right">
        <Segmented
          variant="hud"
          label="Layout"
          value={mode}
          onChange={onModeChange}
          options={[
            {
              id: "page",
              label: "Page",
              icon: "page",
              hint: "One view at a time, with a companion drawer",
              testId: "layout-page",
            },
            {
              id: "split",
              label: "Split",
              icon: "split",
              hint: "Two views side by side",
              testId: "layout-split",
            },
            {
              id: "all",
              label: "All",
              icon: "all",
              hint: "All three views side by side",
              testId: "layout-all",
            },
          ]}
        />
        <Segmented<ThemeChoice>
          variant="hud"
          label="Theme"
          value={theme}
          onChange={setTheme}
          options={[
            { id: "light", label: "", icon: "sun", hint: "Light", testId: "theme-light" },
            { id: "dark", label: "", icon: "moon", hint: "Dark", testId: "theme-dark" },
            { id: "auto", label: "", icon: "auto", hint: "Follow the system", testId: "theme-auto" },
          ]}
        />
        <ConnectionBadge status={status} />
      </div>
    </header>
  );
}
