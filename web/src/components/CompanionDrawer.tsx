import { viewInfo, type ViewId } from "../nav/views";
import { useLayout } from "../state/layout";
import { EnvironmentView } from "../views/environment/EnvironmentView";
import { Standings } from "../views/environment/Standings";
import { NetworkView } from "../views/network/NetworkView";
import { Segmented } from "./ui/controls";
import { Icon } from "./ui/Icon";
import { ViewPanel } from "./ViewPanel";

/**
 * The collapsible side drawer on each page: a compact version of a linked view, so the user can
 * see cause and effect together (for example the network beside the track) without a split.
 * Open, it is a panel with a header; closed, a thin rail whose whole length is the button that
 * opens it again (task S1). The header has exactly one button, the toggle; on the Environment page
 * it also has two tabs, the network and the standings of the cars (task E3), remembered per browser.
 */
export function CompanionDrawer({
  page,
  open,
  onToggle,
}: {
  page: ViewId;
  open: boolean;
  onToggle: () => void;
}) {
  const companion = viewInfo(viewInfo(page).companion);
  const tab = useLayout((s) => s.drawerTab);
  const setTab = useLayout((s) => s.setDrawerTab);
  const tabbed = page === "environment" && companion.id === "network";
  const standings = tabbed && tab === "standings";
  const drawerId = `drawer-${page}`;
  const hint = open ? `Hide ${companion.title}` : `Show ${companion.title}`;
  return (
    <aside className="drawer" data-open={open} data-testid="companion-drawer">
      <div className="drawer-head">
        <button
          type="button"
          className="drawer-toggle"
          aria-expanded={open}
          aria-controls={drawerId}
          aria-label={hint}
          title={hint}
          onClick={onToggle}
        >
          <Icon name={open ? "right" : "left"} />
          {!open && <span className="drawer-vertical">{companion.title}</span>}
        </button>
        {open && tabbed && (
          <Segmented
            label="Drawer"
            value={tab}
            onChange={setTab}
            options={[
              { id: "network", label: "Network", icon: "network", hint: companion.question, testId: "drawer-tab-network" },
              { id: "standings", label: "Standings", icon: "chart", hint: "Every car in race order", testId: "drawer-tab-standings" },
            ]}
          />
        )}
        {open && !tabbed && (
          <span className="eyebrow drawer-title">
            {companion.title} · {companion.question}
          </span>
        )}
      </div>
      {open && (
        <div id={drawerId} className="drawer-body">
          {standings ? (
            <Standings />
          ) : companion.id === "environment" ? (
            <EnvironmentView compact />
          ) : companion.id === "network" ? (
            <NetworkView compact />
          ) : (
            <ViewPanel title={companion.title} comingIn={companion.comingIn} compact />
          )}
        </div>
      )}
    </aside>
  );
}
