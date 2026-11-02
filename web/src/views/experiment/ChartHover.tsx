import type { ReactNode } from "react";

/**
 * A chart's hover read-out as an HTML box laid over the SVG, not as SVG text. The charts are
 * scaled to fill their panel (about 1.9 times on a wide screen), and SVG text with an outline
 * scaled that far smears; HTML text keeps its own size and stays sharp. The same box as
 * "Compare runs" (task 4.7b follow-up); used by the loss chart, the prediction plot and the
 * Training chart since the task 5.8 follow-up.
 *
 * `x` and `y` are in the chart's viewBox units (`width` × `height`); the box sits to the right of
 * `x`, or to its left in the right half so it never leaves the chart. The parent must be a
 * `.cmp-plot` element (positioned) that holds the SVG.
 */
export function ChartHover({
  x,
  y,
  width,
  height,
  testId,
  children,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  testId?: string;
  children: ReactNode;
}) {
  return (
    <div
      className={x > width / 2 ? "cmp-hover cmp-hover-left" : "cmp-hover"}
      data-testid={testId}
      style={{ left: `${(x / width) * 100}%`, top: `${(y / height) * 100}%` }}
    >
      {children}
    </div>
  );
}
