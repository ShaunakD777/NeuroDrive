import type { ReactNode } from "react";

/** A placeholder card for one of the three views, until the real view is built. */
export function ViewPanel({
  title,
  comingIn,
  compact = false,
  headerExtra,
}: {
  title: string;
  comingIn: string;
  /** The small version shown in a companion drawer. */
  compact?: boolean;
  /** An optional control shown at the right of the header (e.g. Split's view picker). */
  headerExtra?: ReactNode;
}) {
  return (
    <section className={compact ? "view view-compact" : "view"}>
      <div className="view-header">
        <h2>{compact ? `${title} (compact)` : title}</h2>
        {headerExtra}
      </div>
      <p>
        Placeholder. The real {compact ? "compact view" : "view"} arrives in {comingIn}.
      </p>
    </section>
  );
}
