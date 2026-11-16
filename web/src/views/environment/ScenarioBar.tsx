import { useEffect, useState } from "react";

import type { PopulationInfo } from "../../api/generated";
import { type ScenarioRequest, useLive } from "../../state/live";
import { useSelection } from "../../state/selection";
import { defaultGhost } from "./population";

// The scenario, in the view header (task E3, recommendation E12): track, driver, number of cars,
// random start and seed, which car is selected, and a raced GA generation's generation, start and
// ghost. Moved from the pre-redesign Controls.tsx with every rule and test id unchanged. Changing a
// setting asks the server to load a new scenario, which starts paused.

const MAX_CARS = 100; // the server's limit for one scenario
const DEFAULT_SEED = 1;

/** A run id without its leading `YYYYMMDD-HHMMSS-` time stamp. */
const shortRun = (id: string) => id.replace(/^\d{8}-\d{6}-/, "");

/**
 * The ghost's generation after moving the race to generation `unit`: one behind if it was one
 * behind (the default), kept if it is still earlier, none if there was none.
 */
export function ghostAfterMove(info: PopulationInfo, unit: number): number | null {
  const ghost = info.ghost_unit;
  if (ghost === null) return null;
  if (ghost === info.unit - 1 || ghost >= unit) return defaultGhost(unit);
  return ghost;
}

/**
 * A whole-number box that applies its value when the user finishes (Enter, leaving the box, or
 * the little arrows) and only if it is a whole number inside the allowed range. Typing a half
 * finished number never reaches the server. It shows `value` again whenever that changes.
 */
export function IntField({
  label,
  value,
  min,
  max,
  disabled,
  testId,
  title,
  onCommit,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  disabled: boolean;
  testId: string;
  title?: string;
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
    <label className="control" title={title}>
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

/** The scenario settings and the car picker, as one wrapping row in the view header. */
export function ScenarioBar() {
  const loadScenario = useLive((s) => s.loadScenario);
  const tracks = useLive((s) => s.tracks);
  const scenario = useLive((s) => s.scenario);
  const frame = useLive((s) => s.frame);
  const connected = useLive((s) => s.status === "connected");
  const isRecording = useLive((s) => s.recording !== null);

  const ready = connected && scenario !== null && frame !== null;
  const nCars = scenario?.n_cars ?? 1;
  const seed = scenario?.seed ?? null;
  // A replayed checkpoint (task 5.9) always drives the exam's 10 starts: cars and start are fixed.
  const replay = scenario?.checkpoint ?? null;
  // A raced GA generation (task 7.6): one car per network, on the run's own track and start.
  const race = scenario?.population ?? null;
  const fixedCars = replay !== null || race !== null;
  const startShift = scenario?.start_shift ?? null;
  const selected = frame?.selected?.index ?? 0;

  return (
    <div className="scenario-bar" role="toolbar" aria-label="Scenario" data-testid="scenario-bar">
      <label className="control">
        <span className="control-label">Track</span>
        <select
          data-testid="select-track"
          value={scenario?.track ?? ""}
          disabled={!connected || tracks.length === 0 || isRecording || race !== null}
          title={race !== null ? "A GA generation races on the track its fitness was measured on" : undefined}
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
          disabled={!ready || isRecording}
          title="Who drives: you, a fixed rule, one neuron you wire, random networks, the scripted expert"
          onChange={(e) => {
            loadScenario({ policy: e.target.value as ScenarioRequest["policy"] });
            e.target.blur();
          }}
        >
          <option value="keyboard">Keyboard (you drive)</option>
          <option value="constant">Constant (steer 0, throttle 0.3)</option>
          <option value="neuron">Neuron (you set its weights)</option>
          <option value="random_networks">Random networks (one per car)</option>
          <option value="expert">Expert (scripted, reads the track)</option>
          {race !== null && (
            <option value="population">
              GA generation: {shortRun(race.run)} · generation {race.unit}
            </option>
          )}
          {replay !== null && (
            <option value="checkpoint">
              Checkpoint: {shortRun(replay.run)} · unit {replay.unit} ({startShift === null ? "exam starts" : "shifted starts"})
            </option>
          )}
        </select>
      </label>

      <IntField
        label="Cars"
        testId="input-cars"
        value={nCars}
        min={1}
        max={MAX_CARS}
        disabled={!ready || fixedCars || isRecording}
        title={fixedCars ? "Fixed: one car per network, or the exam's 10 starts" : "How many cars drive at once (1 to 100)"}
        onCommit={(n) => loadScenario({ n_cars: n })}
      />

      <label className="control control-check" title="Start each car at a seeded random place instead of the start line">
        <input
          type="checkbox"
          data-testid="check-random-start"
          checked={seed !== null}
          disabled={!ready || fixedCars || isRecording}
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
          disabled={!ready || isRecording}
          onCommit={(n) => loadScenario({ seed: n })}
        />
      )}

      {nCars > 1 && (
        <label className="control" title="The car the rays, the trail, the telemetry and the Network view follow">
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
                {race !== null && i === race.ghost_car
                  ? `Ghost (generation ${race.ghost_unit}'s best)`
                  : race !== null
                    ? `Car ${i + 1} (rank ${race.ranks[i] + 1})`
                    : `Car ${i + 1}`}
              </option>
            ))}
          </select>
        </label>
      )}

      {race !== null && (
        <span className="scenario-group" data-testid="race-controls">
          <IntField
            label="Generation"
            testId="race-generation"
            value={race.unit}
            min={1}
            max={race.last_unit}
            disabled={!ready}
            onCommit={(n) =>
              loadScenario({ population: { run: race.run, unit: n, start: race.start, ghost_unit: ghostAfterMove(race, n) } })
            }
          />
          <span className="camera-note">of {race.last_unit}</span>
          <IntField
            label="Start"
            testId="race-start"
            value={race.start + 1}
            min={1}
            max={race.n_starts}
            disabled={!ready}
            onCommit={(n) =>
              loadScenario({ population: { run: race.run, unit: race.unit, start: n - 1, ghost_unit: race.ghost_unit } })
            }
          />
          <span className="camera-note" data-testid="race-start-place">
            {race.start_s0_m.toFixed(0)} m along, {race.start_lateral_m.toFixed(1)} m left
          </span>
          <IntField
            label="Ghost: generation (0 = none)"
            testId="race-ghost"
            value={race.ghost_unit ?? 0}
            min={0}
            max={race.unit - 1}
            disabled={!ready || race.unit === 1}
            title="An earlier generation's champion, see-through, racing alongside"
            onCommit={(n) =>
              loadScenario({ population: { run: race.run, unit: race.unit, start: race.start, ghost_unit: n === 0 ? null : n } })
            }
          />
        </span>
      )}
    </div>
  );
}
