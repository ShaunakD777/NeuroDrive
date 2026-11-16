import { useLive } from "../../state/live";
import { useSelection } from "../../state/selection";
import { bodyCss, bodyOf } from "./cars";
import { minimapFrame } from "./stageLayout";

/**
 * The minimap (task E3, recommendation E13): the whole circuit with every car as a dot in its body
 * colour, shown while the camera is close in, so "where on the lap is this?" needs no zooming out.
 * Stopped cars are faded; the selected car is ringed in the selection colour. Clicking a dot
 * selects that car (the one way to change the car, `selectCar`).
 */
export function Minimap() {
  const geometry = useLive((s) => s.geometry);
  const frame = useLive((s) => s.frame);
  const population = useLive((s) => s.scenario?.population ?? null);
  if (geometry === null || frame === null) return null;
  const box = minimapFrame(geometry);
  const { x, y, status } = frame.cars;
  const selected = frame.selected?.index ?? 0;
  const pad = 14;
  const dot = Math.max(4, Math.max(box.w, box.h) / 60);
  const select = (i: number) => useSelection.getState().selectCar(i);
  return (
    <div className="minimap" data-testid="minimap">
      <svg viewBox={`${-pad} ${-pad} ${box.w + 2 * pad} ${box.h + 2 * pad}`} role="img" aria-label="Map of the whole track">
        <path d={box.road} className="minimap-road" strokeWidth={geometry.width} />
        <path d={box.road} className="minimap-centre" />
        {x.map((cx, i) =>
          i === selected ? null : (
            <circle
              key={i}
              cx={cx - box.minX}
              cy={box.maxY - y[i]}
              r={dot}
              className="minimap-car"
              style={{ fill: bodyCss(bodyOf(population, x.length, i, selected)) }}
              opacity={status[i] === 0 ? 1 : 0.4}
              onClick={() => select(i)}
            >
              <title>Car {i + 1}</title>
            </circle>
          ),
        )}
        {selected < x.length && (
          <circle cx={x[selected] - box.minX} cy={box.maxY - y[selected]} r={dot * 1.8} className="minimap-selected" />
        )}
      </svg>
    </div>
  );
}
