// Pure helpers for drawing a network: where the nodes go and how weights and activations are
// styled. No React and no network maths here (the server computes every number; this only maps
// numbers to pixels, colours and words). See docs/ARCHITECTURE.md §8.

export interface Point {
  x: number;
  y: number;
}

export interface Frame {
  width: number;
  height: number;
  /** Space kept free on the left and right of the columns, for the node labels. */
  marginX: number;
  marginY: number;
}

/**
 * Node positions, one array per layer: layers are columns spread evenly from the left margin to
 * the right margin, and the nodes of a layer are spread evenly down the column (a short column
 * is centred, not stretched, so the widest layer decides the spacing).
 */
export function layoutNodes(layerSizes: number[], frame: Frame): Point[][] {
  const columns = layerSizes.length;
  const widest = Math.max(...layerSizes);
  const usableW = frame.width - 2 * frame.marginX;
  const usableH = frame.height - 2 * frame.marginY;
  const gap = widest > 1 ? usableH / (widest - 1) : 0;
  return layerSizes.map((size, layer) => {
    const x = columns > 1 ? frame.marginX + (usableW * layer) / (columns - 1) : frame.width / 2;
    const top = frame.height / 2 - (gap * (size - 1)) / 2;
    return Array.from({ length: size }, (_, i) => ({ x, y: top + gap * i }));
  });
}

/** Weights at or beyond this size are drawn at full thickness and strength. */
export const WEIGHT_FULL = 1;

/** Line thickness for a weight: thin for small |w|, up to 4.3 px at |w| >= 1. */
export function edgeWidth(weight: number): number {
  return 0.3 + 4 * Math.min(1, Math.abs(weight) / WEIGHT_FULL);
}

/** Line opacity for a weight: faint for small |w| so the strong connections stand out. */
export function edgeOpacity(weight: number): number {
  return 0.15 + 0.75 * Math.min(1, Math.abs(weight) / WEIGHT_FULL);
}

export type Sign = "pos" | "neg";

/** Colour family of a weight (positive and negative differ in colour as well as in position). */
export function signOf(x: number): Sign {
  return x < 0 ? "neg" : "pos";
}

/**
 * How a node is filled for the activation `a`: the colour says the sign and the opacity says the
 * strength, |a| clipped to 1 (so tanh outputs and 0..1 readings use the full range, and larger
 * values such as ReLU outputs just saturate). A value of 0 is almost empty.
 */
export function nodeFill(a: number): { sign: Sign; opacity: number } {
  return { sign: signOf(a), opacity: Math.max(0.04, Math.min(1, Math.abs(a))) };
}

/** A number with a real minus sign and a fixed number of decimals. */
export function num(x: number, digits = 2): string {
  const text = Math.abs(x).toFixed(digits);
  return x < 0 && Number(text) !== 0 ? `−${text}` : text;
}

/** Words for a throttle command in [-1, 1]: positive accelerates, negative brakes. */
export function describeThrottle(output: number): string {
  if (output >= 0.7) return "full throttle";
  if (output >= 0.05) return "accelerating";
  if (output > -0.05) return "coasting";
  if (output > -0.7) return "braking";
  return "hard braking";
}

/** The labels of the seven inputs (observation v1), top to bottom. */
export const INPUT_LABELS = [
  "Ray +60°",
  "Ray +30°",
  "Ray 0°",
  "Ray −30°",
  "Ray −60°",
  "Speed",
  "Heading error",
];

/** The labels of the two outputs (action v1), top to bottom. */
export const OUTPUT_LABELS = ["Steer", "Throttle"];

/** The label of a node: input and output nodes have names, hidden ones are numbered from 1. */
export function nodeLabel(layer: number, index: number, layers: number): string {
  if (layer === 0) return INPUT_LABELS[index] ?? `Input ${index + 1}`;
  if (layer === layers - 1) return OUTPUT_LABELS[index] ?? `Output ${index + 1}`;
  return `Hidden ${layer}, neuron ${index + 1}`;
}

/** The title of a column: Inputs, Hidden 1, Hidden 2, ..., Outputs, with the neuron count. */
export function columnTitle(layer: number, layers: number, size: number): string {
  if (layer === 0) return `Inputs (${size})`;
  if (layer === layers - 1) return `Outputs (${size})`;
  return `Hidden ${layer} (${size})`;
}

/** The exact values of a node, for its tooltip and the readout (z and bias exist for layers > 0). */
export function nodeDescription(
  layer: number,
  index: number,
  layers: number,
  a: number,
  z: number | null,
  bias: number | null,
  activation: string | null,
): string {
  const name = nodeLabel(layer, index, layers);
  if (z === null || bias === null || activation === null) return `${name}: value ${num(a, 4)}`;
  return `${name}: z = ${num(z, 4)} (includes bias ${num(bias, 4)}), ${activation}(z) = ${num(a, 4)}`;
}

/** The exact value of a connection, for its tooltip and the readout. */
export function edgeDescription(layer: number, from: number, to: number, layers: number, w: number): string {
  const source = nodeLabel(layer, from, layers);
  const target = nodeLabel(layer + 1, to, layers);
  return `${source} → ${target}: weight ${num(w, 4)}`;
}
