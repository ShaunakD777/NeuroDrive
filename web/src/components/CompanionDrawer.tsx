import { viewInfo, type ViewId } from "../nav/views";
import { EnvironmentView } from "../views/environment/EnvironmentView";
import { NetworkView } from "../views/network/NetworkView";
import { Icon } from "./ui/Icon";
import { ViewPanel } from "./ViewPanel";

/**
 * The collapsible side drawer on each page: a compact version of a linked view, so the user can
 * see cause and effect together (for example the network beside the track) without a split.
 * Open, it is a panel with a header; closed, a thin rail whose whole length is the button that
 * opens it again (task S1). The header has exactly one button, the toggle.
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
        {open && (
          <span className="eyebrow drawer-title">
            {companion.title} · {companion.question}
          </span>
        )}
      </div>
      {open && (
        <div id={drawerId} className="drawer-body">
          {companion.id === "environment" ? (
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
