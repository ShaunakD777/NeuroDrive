import { useLive } from "../../state/live";
import { useMoment } from "../../state/playback";
import { useSelection } from "../../state/selection";
import { describeOutput, sliderRange } from "./inspect";
import { INPUT_LABELS, num, OUTPUT_LABELS } from "./networkLayout";

/**
 * The what-if panel (task 2.10): while paused, change the selected car's inputs and ask the
 * server what its network *would* output. Each slider move sends `what_if` with the current
 * observation and the changed inputs; the answer shows the real output now, the what-if output,
 * and what the sensitivity predicted (a straight-line guess). The simulation never changes.
 */
export function WhatIfPanel() {
  const frame = useLive((s) => s.frame);
  const result = useLive((s) => s.whatIfResult);
  const send = useLive((s) => s.send);
  const overrides = useSelection((s) => s.whatIf);
  const setWhatIf = useSelection((s) => s.setWhatIf);
  const clearWhatIf = useSelection((s) => s.clearWhatIf);
  const moment = useMoment();

  const selected = frame?.selected ?? null;
  if (frame === null || selected === null) return null;
  if (moment !== null) {
    return (
      <section className="whatif" data-testid="whatif-panel">
        <h4>What if?</h4>
        <p className="hint" data-testid="whatif-past-hint">
          What-if asks about the car now. Go back to now to try changing its inputs.
        </p>
      </section>
    );
  }
  if (frame.playing) {
    return (
      <section className="whatif" data-testid="whatif-panel">
        <h4>What if?</h4>
        <p className="hint" data-testid="whatif-paused-hint">
          Pause the simulation to try changing this car's inputs.
        </p>
      </section>
    );
  }

  const real = selected.observation;
  // Only the changed inputs travel; the server keeps the car's exact values for the rest.
  const ask = (next: Record<number, number>) => send({ type: "what_if", overrides: next });
  const changed = Object.keys(overrides).length > 0;
  const showResult = changed && result !== null && result.car === selected.index && result.step === frame.step;

  return (
    <section className="whatif" data-testid="whatif-panel">
      <div className="whatif-head">
        <h4>What if? (car {selected.index + 1}, step {frame.step})</h4>
        <button
          type="button"
          data-testid="whatif-reset"
          disabled={!changed}
          onClick={() => clearWhatIf()}
        >
          Back to the real inputs
        </button>
      </div>
      {real.map((value, k) => {
        const { min, max } = sliderRange(k);
        const current = k in overrides ? overrides[k] : value;
        return (
          <label key={k} className="whatif-row">
            <span>{INPUT_LABELS[k] ?? `Input ${k + 1}`}</span>
            <input
              type="range"
              min={min}
              max={max}
              step={0.01}
              value={current}
              data-testid={`whatif-${k}`}
              onChange={(e) => {
                const v = Number(e.target.value);
                setWhatIf(k, v);
                ask({ ...overrides, [k]: v });
              }}
            />
            <span className={k in overrides ? "cnet-value whatif-changed" : "cnet-value"} data-testid={`whatif-value-${k}`}>
              {num(current)}
            </span>
          </label>
        );
      })}

      {showResult ? (
        <table className="whatif-table" data-testid="whatif-result">
          <thead>
            <tr>
              <th />
              <th>now</th>
              <th>what-if</th>
              <th>predicted</th>
            </tr>
          </thead>
          <tbody>
            {OUTPUT_LABELS.map((label, m) => {
              const now = result.base_outputs[m] ?? 0;
              const tried = result.trace.activations[result.trace.activations.length - 1]?.[m] ?? 0;
              const guess = result.predicted_outputs[m] ?? 0;
              return (
                <tr key={label} data-testid={`whatif-out-${m}`} data-now={now} data-tried={tried} data-predicted={guess}>
                  <th>{label}</th>
                  <td>{num(now, 3)}</td>
                  <td>
                    <strong>{num(tried, 3)}</strong> <span className="hint">{describeOutput(m, tried)}</span>
                  </td>
                  <td>{num(guess, 3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        <p className="hint">
          Move a slider: the server runs this car's network on the changed inputs. "Predicted" is
          what the sensitivity says; "what-if" is the network's real answer.
        </p>
      )}
    </section>
  );
}
