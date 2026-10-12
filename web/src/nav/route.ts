import { useEffect, useState } from "react";

import { DEFAULT_VIEW, isViewId, type ViewId } from "./views";

// Client-side routing with the address hash: `#/network` means "show the Network view".
// Changing the hash never reloads the page, so the selection store and the live connection
// survive, and the browser's back and forward buttons move between views for free.

/** The view named by a hash such as "#/network", or null if it names no view. */
export function parseHash(hash: string): ViewId | null {
  const name = hash.replace(/^#\/?/, "").split(/[/?]/)[0].toLowerCase();
  return isViewId(name) ? name : null;
}

export function hashFor(view: ViewId): string {
  return `#/${view}`;
}

/**
 * The view in the address bar, kept up to date. An empty or unknown hash is replaced (without
 * adding a history entry) by the default view, so the address always names a real page.
 */
export function useRoute(): ViewId {
  const [view, setView] = useState<ViewId>(() => parseHash(window.location.hash) ?? DEFAULT_VIEW);

  useEffect(() => {
    function sync(): void {
      const parsed = parseHash(window.location.hash);
      if (parsed === null) {
        window.history.replaceState(null, "", hashFor(DEFAULT_VIEW));
        setView(DEFAULT_VIEW);
      } else {
        setView(parsed);
      }
    }
    sync();
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);

  return view;
}
