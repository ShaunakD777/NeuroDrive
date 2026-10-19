import { describe, expect, it } from "vitest";

import {
  columnTitle,
  edgeDescription,
  edgeOpacity,
  edgeWidth,
  INPUT_LABELS,
  layoutNodes,
  nodeDescription,
  nodeFill,
  nodeLabel,
  num,
  OUTPUT_LABELS,
  signOf,
} from "./networkLayout";

const frame = { width: 800, height: 400, marginX: 100, marginY: 20 };

describe("layoutNodes", () => {
  const nodes = layoutNodes([7, 16, 16, 2], frame);

  it("makes one column per layer with the right number of nodes", () => {
    expect(nodes.map((column) => column.length)).toEqual([7, 16, 16, 2]);
  });

  it("spreads the columns evenly between the margins", () => {
    expect(nodes.map((column) => column[0].x)).toEqual([100, 300, 500, 700]);
  });

  it("keeps every node inside the frame and in a column's order top to bottom", () => {
    for (const column of nodes) {
      for (const p of column) {
        expect(p.y).toBeGreaterThanOrEqual(20);
        expect(p.y).toBeLessThanOrEqual(380);
      }
      for (let i = 1; i < column.length; i++) expect(column[i].y).toBeGreaterThan(column[i - 1].y);
    }
  });

  it("uses the full height for the widest layer and centres the shorter ones", () => {
    expect(nodes[1][0].y).toBeCloseTo(20);
    expect(nodes[1][15].y).toBeCloseTo(380);
    const gap = (380 - 20) / 15;
    expect(nodes[3][0].y).toBeCloseTo(200 - gap / 2); // 2 nodes around the middle line
    expect(nodes[3][1].y).toBeCloseTo(200 + gap / 2);
    const mean = (ys: number[]) => ys.reduce((a, b) => a + b, 0) / ys.length;
    expect(mean(nodes[0].map((p) => p.y))).toBeCloseTo(200);
  });

  it("handles a single node and a single column", () => {
    expect(layoutNodes([1], frame)).toEqual([[{ x: 400, y: 200 }]]);
    expect(layoutNodes([1, 1], frame)[1][0]).toEqual({ x: 700, y: 200 });
  });
});

describe("weight styling", () => {
  it("grows with |w| and does not care about the sign", () => {
    expect(edgeWidth(0)).toBeCloseTo(0.3);
    expect(edgeWidth(0.5)).toBeCloseTo(2.3);
    expect(edgeWidth(-0.5)).toBe(edgeWidth(0.5));
    expect(edgeWidth(1)).toBeCloseTo(4.3);
    expect(edgeWidth(5)).toBe(edgeWidth(1)); // saturates
    expect(edgeOpacity(0)).toBeCloseTo(0.15);
    expect(edgeOpacity(-1)).toBeCloseTo(0.9);
    expect(edgeOpacity(0.2)).toBeLessThan(edgeOpacity(0.6));
  });

  it("colours by sign", () => {
    expect(signOf(0.3)).toBe("pos");
    expect(signOf(-0.3)).toBe("neg");
    expect(signOf(0)).toBe("pos");
  });
});

describe("nodeFill", () => {
  it("has the sign of the activation and the strength of its size", () => {
    expect(nodeFill(0.5)).toEqual({ sign: "pos", opacity: 0.5 });
    expect(nodeFill(-0.905148)).toEqual({ sign: "neg", opacity: 0.905148 });
  });

  it("saturates at 1 and never disappears completely", () => {
    expect(nodeFill(7).opacity).toBe(1);
    expect(nodeFill(-7).opacity).toBe(1);
    expect(nodeFill(0).opacity).toBe(0.04);
  });
});

describe("labels and numbers", () => {
  it("formats numbers with a real minus and hides a negative zero", () => {
    expect(num(-1.5)).toBe("−1.50");
    expect(num(0.2, 4)).toBe("0.2000");
    expect(num(-0.0001)).toBe("0.00");
  });

  it("names the inputs, the outputs and the hidden neurons", () => {
    expect(INPUT_LABELS).toHaveLength(7);
    expect(OUTPUT_LABELS).toEqual(["Steer", "Throttle"]);
    expect(nodeLabel(0, 2, 4)).toBe("Ray 0°");
    expect(nodeLabel(0, 6, 4)).toBe("Heading error");
    expect(nodeLabel(3, 1, 4)).toBe("Throttle");
    expect(nodeLabel(2, 4, 4)).toBe("Hidden 2, neuron 5");
  });

  it("titles the columns", () => {
    expect([0, 1, 2, 3].map((l) => columnTitle(l, 4, l === 0 ? 7 : l === 3 ? 2 : 16))).toEqual([
      "Inputs (7)",
      "Hidden 1 (16)",
      "Hidden 2 (16)",
      "Outputs (2)",
    ]);
  });

  it("describes a node and a connection with their exact values", () => {
    expect(nodeDescription(0, 5, 4, 0.5, null, null, null)).toBe("Speed: value 0.5000");
    expect(nodeDescription(1, 0, 4, -0.905148, -1.5, 0, "tanh")).toBe(
      "Hidden 1, neuron 1: z = −1.5000 (includes bias 0.0000), tanh(z) = −0.9051",
    );
    expect(edgeDescription(0, 1, 0, 4, -1)).toBe("Ray +30° → Hidden 1, neuron 1: weight −1.0000");
    expect(edgeDescription(2, 3, 1, 4, 0.25)).toBe("Hidden 2, neuron 4 → Throttle: weight 0.2500");
  });
});
