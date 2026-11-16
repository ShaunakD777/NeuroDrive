import { create } from "zustand";

import { isViewId, VIEW_IDS, type ViewId } from "../nav/views";

// How the screen is arranged (docs/ARCHITECTURE.md §8.0). This is separate from the selection
// store on purpose: selection is *what* the user picked, layout is *how* the UI is arranged.
// The choice is remembered in this browser's local storage; if storage is unavailable (private
// window, blocked site data) the app silently uses the defaults.

export type LayoutMode = "page" | "split" | "all";

/** The theme setting: always light, always dark, or follow the operating system ("auto"). */
export type ThemeChoice = "light" | "dark" | "auto";

export interface LayoutState {
  mode: LayoutMode;
  /** The view shown next to the current one in Split mode. */
  splitPartner: ViewId;
  /** Whether each page's companion drawer is open. */
  drawerOpen: Record<ViewId, boolean>;
  /** Light, dark or follow the system (task S1). */
  theme: ThemeChoice;
}

export interface LayoutActions {
  setMode: (mode: LayoutMode) => void;
  setSplitPartner: (view: ViewId) => void;
  toggleDrawer: (view: ViewId) => void;
  setTheme: (theme: ThemeChoice) => void;
}

export const STORAGE_KEY = "neurodrive.layout.v1";

export const defaultLayout: LayoutState = {
  mode: "page",
  splitPartner: "network",
  drawerOpen: { environment: true, network: true, experiment: true },
  theme: "auto",
};

const LAYOUT_MODES: readonly LayoutMode[] = ["page", "split", "all"];
const THEMES: readonly ThemeChoice[] = ["light", "dark", "auto"];

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
    theme: THEMES.includes(saved.theme as ThemeChoice) ? (saved.theme as ThemeChoice) : defaultLayout.theme,
  };
}

/**
 * The theme actually shown: the setting itself when it is "light" or "dark"; for "auto", dark
 * exactly when the operating system asks for dark.
 */
export function resolveTheme(choice: ThemeChoice, systemIsDark: boolean): "light" | "dark" {
  if (choice !== "auto") return choice;
  return systemIsDark ? "dark" : "light";
}

const DARK_QUERY = "(prefers-color-scheme: dark)";

/**
 * Write the resolved theme to `<html data-theme>`, which the stylesheets (tokens.css) and the
 * canvases (they read CSS variables on every draw) follow. Does nothing outside a browser.
 */
export function applyTheme(choice: ThemeChoice = useLayout.getState().theme): void {
  if (typeof document === "undefined") return;
  const systemIsDark = globalThis.matchMedia?.(DARK_QUERY).matches ?? false;
  document.documentElement.dataset.theme = resolveTheme(choice, systemIsDark);
}

/** Call `listener` whenever the operating system switches between light and dark; returns the unsubscribe. */
export function onSystemThemeChange(listener: () => void): () => void {
  const media = globalThis.matchMedia?.(DARK_QUERY);
  if (!media) return () => {};
  media.addEventListener("change", listener);
  return () => media.removeEventListener("change", listener);
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
    const { mode, splitPartner, drawerOpen, theme } = state;
    globalThis.localStorage?.setItem(
      STORAGE_KEY,
      JSON.stringify({ mode, splitPartner, drawerOpen, theme }),
    );
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
  setTheme: (theme) => set({ theme }),
}));

useLayout.subscribe(saveLayout);
