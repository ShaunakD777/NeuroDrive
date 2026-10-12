import { create } from "zustand";

import { isViewId, VIEW_IDS, type ViewId } from "../nav/views";

// How the screen is arranged (docs/ARCHITECTURE.md §8.0). This is separate from the selection
// store on purpose: selection is *what* the user picked, layout is *how* the UI is arranged.
// The choice is remembered in this browser's local storage; if storage is unavailable (private
// window, blocked site data) the app silently uses the defaults.

export type LayoutMode = "page" | "split" | "all";

export interface LayoutState {
  mode: LayoutMode;
  /** The view shown next to the current one in Split mode. */
  splitPartner: ViewId;
  /** Whether each page's companion drawer is open. */
  drawerOpen: Record<ViewId, boolean>;
}

export interface LayoutActions {
  setMode: (mode: LayoutMode) => void;
  setSplitPartner: (view: ViewId) => void;
  toggleDrawer: (view: ViewId) => void;
}

export const STORAGE_KEY = "neurodrive.layout.v1";

export const defaultLayout: LayoutState = {
  mode: "page",
  splitPartner: "network",
  drawerOpen: { environment: true, network: true, experiment: true },
};

const LAYOUT_MODES: readonly LayoutMode[] = ["page", "split", "all"];

/** Keep only well-formed saved values; anything missing or odd falls back to the default. */
export function sanitizeLayout(raw: unknown): LayoutState {
  const saved = (typeof raw === "object" && raw !== null ? raw : {}) as Record<string, unknown>;
  const drawers = (
    typeof saved.drawerOpen === "object" && saved.drawerOpen !== null ? saved.drawerOpen : {}
  ) as Record<string, unknown>;
  const drawerOpen = { ...defaultLayout.drawerOpen };
  for (const id of VIEW_IDS) {
    if (typeof drawers[id] === "boolean") drawerOpen[id] = drawers[id] as boolean;
  }
  return {
    mode: LAYOUT_MODES.includes(saved.mode as LayoutMode)
      ? (saved.mode as LayoutMode)
      : defaultLayout.mode,
    splitPartner: isViewId(saved.splitPartner) ? saved.splitPartner : defaultLayout.splitPartner,
    drawerOpen,
  };
}

/** The partner actually shown in Split mode: never the current view itself. */
export function effectivePartner(current: ViewId, partner: ViewId): ViewId {
  if (partner !== current) return partner;
  return VIEW_IDS[(VIEW_IDS.indexOf(current) + 1) % VIEW_IDS.length];
}

function loadLayout(): LayoutState {
  try {
    const text = globalThis.localStorage?.getItem(STORAGE_KEY);
    return text ? sanitizeLayout(JSON.parse(text)) : { ...defaultLayout };
  } catch {
    return { ...defaultLayout };
  }
}

function saveLayout(state: LayoutState): void {
  try {
    const { mode, splitPartner, drawerOpen } = state;
    globalThis.localStorage?.setItem(STORAGE_KEY, JSON.stringify({ mode, splitPartner, drawerOpen }));
  } catch {
    // Storage unavailable: the layout still works, it just is not remembered.
  }
}

export const useLayout = create<LayoutState & LayoutActions>()((set) => ({
  ...loadLayout(),
  setMode: (mode) => set({ mode }),
  setSplitPartner: (splitPartner) => set({ splitPartner }),
  toggleDrawer: (view) =>
    set((s) => ({ drawerOpen: { ...s.drawerOpen, [view]: !s.drawerOpen[view] } })),
}));

useLayout.subscribe(saveLayout);
