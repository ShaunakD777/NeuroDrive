import { viewInfo, type ViewId } from "../nav/views";
import { ViewPanel } from "./ViewPanel";

/**
 * The collapsible side drawer on each page: a compact version of a linked view, so the user can
 * see cause and effect together (for example the network beside the track) without a split.
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
  return (
    <aside className={open ? "drawer drawer-open" : "drawer"} data-testid="companion-drawer">
      <button
        type="button"
        className="drawer-toggle"
        aria-expanded={open}
        aria-controls={drawerId}
        onClick={onToggle}
        title={open ? `Hide ${companion.title}` : `Show ${companion.title}`}
      >
        <span aria-hidden="true">{open ? "›" : "‹"}</span>
        <span className="drawer-toggle-label">{companion.title}</span>
      </button>
      {open && (
        <div id={drawerId} className="drawer-body">
          <ViewPanel title={companion.title} comingIn={companion.comingIn} compact />
        </div>
      )}
    </aside>
  );
}
