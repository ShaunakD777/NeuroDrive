import { beforeEach, describe, expect, it } from "vitest";

import { defaultLayout, effectivePartner, sanitizeLayout, useLayout } from "./layout";

describe("sanitizeLayout", () => {
  it("falls back to the defaults for missing or broken data", () => {
    expect(sanitizeLayout(null)).toEqual(defaultLayout);
    expect(sanitizeLayout("nonsense")).toEqual(defaultLayout);
    expect(sanitizeLayout({ mode: "sideways", splitPartner: "garage" })).toEqual(defaultLayout);
  });

  it("keeps valid saved values and fills in the rest", () => {
    const layout = sanitizeLayout({ mode: "split", drawerOpen: { network: false, bogus: 1 } });
    expect(layout.mode).toBe("split");
    expect(layout.splitPartner).toBe(defaultLayout.splitPartner);
    expect(layout.drawerOpen).toEqual({ environment: true, network: false, experiment: true });
  });
});

describe("effectivePartner", () => {
  it("uses the chosen partner when it differs from the current view", () => {
    expect(effectivePartner("environment", "experiment")).toBe("experiment");
  });

  it("moves on to the next view when the partner equals the current view", () => {
    expect(effectivePartner("environment", "environment")).toBe("network");
    expect(effectivePartner("experiment", "experiment")).toBe("environment");
  });
});

describe("layout store", () => {
  beforeEach(() => {
    useLayout.setState({ ...defaultLayout, drawerOpen: { ...defaultLayout.drawerOpen } });
  });

  // Vitest runs in Node, which has no localStorage: these also show the store works without it.
  it("starts in page mode with every drawer open", () => {
    const s = useLayout.getState();
    expect(s.mode).toBe("page");
    expect(Object.values(s.drawerOpen).every(Boolean)).toBe(true);
  });

  it("changes mode and partner", () => {
    useLayout.getState().setMode("all");
    useLayout.getState().setSplitPartner("experiment");
    expect(useLayout.getState().mode).toBe("all");
    expect(useLayout.getState().splitPartner).toBe("experiment");
  });

  it("toggles one page's drawer without touching the others", () => {
    useLayout.getState().toggleDrawer("network");
    expect(useLayout.getState().drawerOpen).toEqual({
      environment: true,
      network: false,
      experiment: true,
    });
  });
});
