// The three views of NeuroDrive (docs/ARCHITECTURE.md §8.0). Every place that lists, names or
// links the views reads this table, so adding a detail here updates the tabs, pages and drawers.

export type ViewId = "environment" | "network" | "experiment";

export interface ViewInfo {
  id: ViewId;
  title: string;
  /** Which roadmap task replaces the placeholder with the real view. */
  comingIn: string;
  /** The linked view shown compactly in this page's side drawer. */
  companion: ViewId;
}

export const VIEWS: readonly ViewInfo[] = [
  { id: "environment", title: "Environment", comingIn: "task 1.8", companion: "network" },
  {
    id: "network",
    title: "Network",
    comingIn: "task 1.9 and Phase 2",
    companion: "environment",
  },
  {
    id: "experiment",
    title: "Experiment",
    comingIn: "Phase 4 and later",
    companion: "environment",
  },
];

export const VIEW_IDS: readonly ViewId[] = VIEWS.map((v) => v.id);

export const DEFAULT_VIEW: ViewId = "environment";

export function viewInfo(id: ViewId): ViewInfo {
  return VIEWS.find((v) => v.id === id)!;
}

export function isViewId(value: unknown): value is ViewId {
  return typeof value === "string" && (VIEW_IDS as readonly string[]).includes(value);
}
