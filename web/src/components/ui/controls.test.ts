import { createElement as h } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { Button, Chip, Panel, Segmented, Select, Stat, ValueBar } from "./controls";

// The shared controls (task S0) must carry the test ids the browser tests read, on the element
// the tests act on, or a view moved onto them would lose its hooks (prototypes/2d-redesign/MIGRATION_AUDIT.md §7.4).

const noop = () => {};

describe("Select", () => {
  it("puts the test id on the <select> itself, so selectOption and toHaveValue work", () => {
    const html = renderToStaticMarkup(
      h(Select<string>, {
        value: "expert",
        options: [
          { value: "keyboard", label: "Keyboard" },
          { value: "expert", label: "Expert" },
        ],
        onChange: noop,
        testId: "select-policy",
      }),
    );
    expect(html).toMatch(/<select[^>]*data-testid="select-policy"/);
    expect(html).toMatch(/<option value="expert" selected="">Expert<\/option>/);
  });

  it("keeps a number value a number in the markup", () => {
    const html = renderToStaticMarkup(
      h(Select<number>, { value: 8, options: [{ value: 1, label: "1×" }, { value: 8, label: "8×" }], onChange: noop }),
    );
    expect(html).toMatch(/<option value="8" selected="">8×<\/option>/);
  });
});

describe("Segmented", () => {
  it("gives every option its own test id and marks the chosen one pressed", () => {
    const html = renderToStaticMarkup(
      h(Segmented<string>, {
        label: "Camera",
        value: "follow",
        onChange: noop,
        options: [
          { id: "follow", label: "Follow", testId: "camera-follow" },
          { id: "fit", label: "Whole track", testId: "camera-fit" },
          { id: "free", label: "Free" },
        ],
      }),
    );
    expect(html).toMatch(/<button[^>]*aria-pressed="true"[^>]*data-testid="camera-follow"/);
    expect(html).toMatch(/<button[^>]*aria-pressed="false"[^>]*data-testid="camera-fit"/);
    expect(html).toContain('data-testid="seg-free"'); // the default when no test id is given
    expect(html).toContain('role="group" aria-label="Camera"');
  });
});

describe("Button, Panel, Chip, Stat, ValueBar", () => {
  it("Button: the test id, a label when it shows only an icon, and the pressed state", () => {
    const html = renderToStaticMarkup(h(Button, { icon: "play", title: "Play", pressed: false, testId: "btn-play" }));
    expect(html).toMatch(/^<button type="button" class="btn btn-default btn-icon"/);
    expect(html).toContain('aria-label="Play"');
    expect(html).toContain('aria-pressed="false"');
    expect(html).toContain('data-testid="btn-play"');
  });

  it("Panel: the test id on the section", () => {
    const html = renderToStaticMarkup(h(Panel, { title: "Telemetry", testId: "telemetry", children: "body" }));
    expect(html).toMatch(/^<section class="panel "[^>]*data-testid="telemetry"/);
    expect(html).toContain('<h3 class="panel-title">Telemetry</h3>');
  });

  it("Chip: colour is never alone (an icon and the word come with it)", () => {
    const html = renderToStaticMarkup(h(Chip, { kind: "bad", children: "crashed" }));
    expect(html).toContain('data-kind="bad"');
    expect(html).toContain("<svg");
    expect(html).toContain("crashed");
  });

  it("Stat: label, value and unit", () => {
    const html = renderToStaticMarkup(h(Stat, { label: "Speed", value: "12.3", unit: "m/s" }));
    expect(html).toContain('<span class="eyebrow">Speed</span>');
    expect(html).toContain('12.3<span class="stat-unit">m/s</span>');
  });

  it("ValueBar: a signed value grows from the middle, clamped to −1..1", () => {
    const right = renderToStaticMarkup(h(ValueBar, { value: 0.5 }));
    expect(right).toContain("left:50%;width:25%");
    const left = renderToStaticMarkup(h(ValueBar, { value: -2 }));
    expect(left).toContain("left:0%;width:50%");
    expect(left).toContain("vbar-neg");
    const unsigned = renderToStaticMarkup(h(ValueBar, { value: 0.3, signed: false }));
    expect(unsigned).toContain("left:0;width:30%");
  });
});
