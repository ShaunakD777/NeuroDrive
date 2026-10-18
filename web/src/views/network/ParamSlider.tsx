import { useEffect, useState } from "react";

import { useLive } from "../../state/live";

type NumericParam = "w_left" | "w_right" | "bias" | "throttle";

/**
 * A slider plus a number box for one neuron parameter.
 *
 * Moving the slider (or typing a number and pressing Enter) sends `set_param` to the server; the
 * server answers with a frame whose numbers come back as `value`. While the user is dragging, the
 * slider shows its own position so it does not jump around while the server catches up; once the
 * user lets go it shows the server's value again.
 */
export function ParamSlider({
  label,
  name,
  value,
  min,
  max,
  step,
  decimals = 2,
}: {
  label: string;
  name: NumericParam;
  /** The server's current value for this parameter. */
  value: number;
  /** The range of the slider (the number box also accepts values the server allows). */
  min: number;
  max: number;
  step: number;
  decimals?: number;
}) {
  const send = useLive((s) => s.send);
  const [dragging, setDragging] = useState<number | null>(null);
  const [text, setText] = useState(value.toFixed(decimals));
  const shown = dragging ?? value;

  useEffect(() => {
    if (dragging === null) setText(value.toFixed(decimals));
  }, [value, dragging, decimals]);

  const push = (v: number) => send({ type: "set_param", name, value: v });

  const commitText = () => {
    const n = Number(text);
    if (text.trim() !== "" && Number.isFinite(n)) {
      if (n !== value) push(n);
    } else {
      setText(value.toFixed(decimals));
    }
  };

  return (
    <div className="param">
      <span className="param-label">{label}</span>
      <input
        type="range"
        className="param-slider"
        data-testid={`slider-${name}`}
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={Math.min(max, Math.max(min, shown))}
        onChange={(e) => {
          const v = Number(e.target.value);
          setDragging(v);
          setText(v.toFixed(decimals));
          push(v);
        }}
        onPointerUp={() => setDragging(null)}
        onKeyUp={() => setDragging(null)}
        onBlur={() => setDragging(null)}
      />
      <input
        type="number"
        className="param-number"
        data-testid={`number-${name}`}
        aria-label={`${label} (number)`}
        step={step}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={commitText}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            commitText();
            e.currentTarget.blur(); // hand the arrow keys back to driving
          }
        }}
      />
    </div>
  );
}
