import { useEffect, useState } from "react";

import { type ScenarioRequest, useLive } from "../../state/live";
import { useSelection } from "../../state/selection";

const SPEEDS = [0.5, 1, 2, 4, 8];
const MAX_CARS = 100; // the server's limit for one scenario
const DEFAULT_SEED = 1;

/**
 * A whole-number box that applies its value when the user finishes (Enter, leaving the box, or
 * the little arrows) and only if it is a whole number inside the allowed range. Typing a half
 * finished number never reaches the server. It shows `value` again whenever that changes.
 */
function IntField({
  label,
  value,
  min,
  max,
  disabled,
  testId,
  onCommit,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  disabled: boolean;
  testId: string;
  onCommit: (value: number) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);

  const commit = () => {
    const n = Number(text);
    if (text.trim() !== "" && Number.isInteger(n) && n >= min && n <= max) {
      if (n !== value) onCommit(n);
    } else {
      setText(String(value)); // not a usable number: show the real one again
    }
  };

  return (
    <label className="control">
      <span className="control-label">{label}</span>
      <input
        type="number"
        className="control-number"
        data-testid={testId}
        min={min}
        max={max}
        step={1}
        value={text}
        disabled={disabled}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            commit();
            e.currentTarget.blur(); // hand the arrow keys back to driving
          }
        }}
      />
    </label>
  );
}

/**
 * Everything that sets up and runs the simulation: the scenario (track, driver, number of cars,
 * start), play / pause / step / reset, the speed factor, which car is selected, and the camera.
 * Changing a scenario setting asks the server to load a new scenario, which starts paused.
 */
export function Controls({
  mode,
  onModeChange,
  onZoomIn,
  onZoomOut,
}: {
  mode: "follow" | "fit" | "free";
  onModeChange: (mode: "follow" | "fit") => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
}) {
  const send = useLive((s) => s.send);
  const loadScenario = useLive((s) => s.loadScenario);
  const tracks = useLive((s) => s.tracks);
  const scenario = useLive((s) => s.scenario);
  const frame = useLive((s) => s.frame);
  const ended = useLive((s) => s.episodeEnd !== null);
  const connected = useLive((s) => s.status === "connected");

  const ready = connected && scenario !== null && frame !== null;
  const playing = frame?.playing ?? false;
  const speed = frame?.speed ?? 1;
  const nCars = scenario?.n_cars ?? 1;
  const seed = scenario?.seed ?? null;
  const selected = frame?.selected?.index ?? 0;

  return (
    <div className="controls" role="toolbar" aria-label="Simulation controls">
      <div className="controls-row">
        <label className="control">
          <span className="control-label">Track</span>
          <select
            data-testid="select-track"
            value={scenario?.track ?? ""}
            disabled={!connected || tracks.length === 0}
            onChange={(e) => {
              loadScenario({ track: e.target.value });
              e.target.blur(); // so the arrow keys drive the car instead of changing this menu
            }}
          >
            {scenario === null && <option value="">…</option>}
            {tracks.map((t) => (
              <option key={t.name} value={t.name}>
                {t.name}
              </option>
            ))}
          </select>
        </label>

        <label className="control">
          <span className="control-label">Driver</span>
          <select
            data-testid="select-policy"
            value={scenario?.policy ?? "keyboard"}
            disabled={!ready}
            onChange={(e) => {
              loadScenario({ policy: e.target.value as ScenarioRequest["policy"] });
              e.target.blur();
            }}
          >
            <option value="keyboard">Keyboard (you drive)</option>
            <option value="constant">Constant (steer 0, throttle 0.3)</option>
            <option value="neuron">Neuron (you set its weights)</option>
            <option value="random_networks">Random networks (one per car)</option>
          </select>
        </label>

        <IntField
          label="Cars"
          testId="input-cars"
          value={nCars}
          min={1}
          max={MAX_CARS}
          disabled={!ready}
          onCommit={(n) => loadScenario({ n_cars: n })}
        />

        <label className="control">
          <input
            type="checkbox"
            data-testid="check-random-start"
            checked={seed !== null}
            disabled={!ready}
            onChange={(e) => {
              loadScenario({ seed: e.target.checked ? DEFAULT_SEED : null });
              e.target.blur();
            }}
          />
          <span className="control-label">Random start</span>
        </label>
        {seed !== null && (
          <IntField
            label="Seed"
            testId="input-seed"
            value={seed}
            min={0}
            max={999_999}
            disabled={!ready}
            onCommit={(n) => loadScenario({ seed: n })}
          />
        )}
      </div>

      <div className="controls-row">
        <button
          type="button"
          data-testid="btn-play"
          disabled={!ready || ended}
          onClick={() => send({ type: playing ? "pause" : "play" })}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          data-testid="btn-step"
          disabled={!ready || ended || playing}
          onClick={() => send({ type: "step" })}
        >
          Step
        </button>
        <button
          type="button"
          data-testid="btn-reset"
          disabled={!ready}
          onClick={() => send({ type: "reset" })}
        >
          Reset
        </button>

        <label className="control">
          <span className="control-label">Speed</span>
          <select
            data-testid="select-speed"
            value={speed}
            disabled={!ready}
            onChange={(e) => {
              send({ type: "set_speed", speed: Number(e.target.value) });
              e.target.blur();
            }}
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </label>

        {nCars > 1 && (
          <label className="control">
            <span className="control-label">Selected car</span>
            <select
              data-testid="select-car"
              value={selected}
              disabled={!ready}
              onChange={(e) => {
                useSelection.getState().selectCar(Number(e.target.value));
                e.target.blur();
              }}
            >
              {Array.from({ length: nCars }, (_, i) => (
                <option key={i} value={i}>
                  Car {i + 1}
                </option>
              ))}
            </select>
          </label>
        )}

        <div className="segmented" role="group" aria-label="Camera">
          <button
            type="button"
            aria-pressed={mode === "follow"}
            data-testid="camera-follow"
            onClick={() => onModeChange("follow")}
          >
            Follow car
          </button>
          <button
            type="button"
            aria-pressed={mode === "fit"}
            data-testid="camera-fit"
            onClick={() => onModeChange("fit")}
          >
            Whole track
          </button>
        </div>
        {mode === "free" && (
          <span className="camera-note" data-testid="camera-free">
            Free view
          </span>
        )}
        <button
          type="button"
          data-testid="zoom-out"
          aria-label="Zoom out"
          onClick={onZoomOut}
        >
          −
        </button>
        <button
          type="button"
          data-testid="zoom-in"
          aria-label="Zoom in"
          onClick={onZoomIn}
        >
          +
        </button>
      </div>
    </div>
  );
}
