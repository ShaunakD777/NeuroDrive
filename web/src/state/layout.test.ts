import { beforeEach, describe, expect, it } from "vitest";

import { defaultLayout, effectivePartner, resolveTheme, sanitizeLayout, useLayout } from "./layout";

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

  it("keeps a saved theme and falls back to following the system", () => {
    expect(defaultLayout.theme).toBe("auto");
    expect(sanitizeLayout({ theme: "dark" }).theme).toBe("dark");
    expect(sanitizeLayout({ theme: "light" }).theme).toBe("light");
    expect(sanitizeLayout({ theme: "sepia" }).theme).toBe("auto");
    expect(sanitizeLayout({ theme: 1 }).theme).toBe("auto");
  });

  it("keeps a saved drawer tab and falls back to the network (task E3)", () => {
    expect(defaultLayout.drawerTab).toBe("network");
    expect(sanitizeLayout({ drawerTab: "standings" }).drawerTab).toBe("standings");
    expect(sanitizeLayout({ drawerTab: "charts" }).drawerTab).toBe("network");
    expect(sanitizeLayout({}).drawerTab).toBe("network");
  });
});

describe("resolveTheme", () => {
  it("uses an explicit choice whatever the system says", () => {
    expect(resolveTheme("light", true)).toBe("light");
    expect(resolveTheme("dark", false)).toBe("dark");
  });

  it("follows the system on auto", () => {
    expect(resolveTheme("auto", true)).toBe("dark");
    expect(resolveTheme("auto", false)).toBe("light");
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

  it("changes the Environment drawer's tab", () => {
    useLayout.getState().setDrawerTab("standings");
    expect(useLayout.getState().drawerTab).toBe("standings");
    useLayout.getState().setDrawerTab("network");
    expect(useLayout.getState().drawerTab).toBe("network");
  });

  it("changes the theme setting", () => {
    useLayout.getState().setTheme("dark");
    expect(useLayout.getState().theme).toBe("dark");
    useLayout.getState().setTheme("auto");
    expect(useLayout.getState().theme).toBe("auto");
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
