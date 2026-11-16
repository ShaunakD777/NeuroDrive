import { useState } from "react";

import { Button } from "../../components/ui/controls";
import { useLive } from "../../state/live";

// Recording keyboard demonstrations (task 6.1b, ADR-051) as a second floating bar under the
// transport, shown for the keyboard driver only (task E3, PRODUCTION_MAPPING §3.2). Moved from the
// pre-redesign Controls.tsx with the same rules, words and test ids.

const DEFAULT_RECORD_TAG = "default";

export function RecordBar() {
  const send = useLive((s) => s.send);
  const ready = useLive((s) => s.status === "connected" && s.scenario !== null && s.frame !== null);
  const recording = useLive((s) => s.recording);
  const savedDataset = useLive((s) => s.savedDataset);
  const [tag, setTag] = useState(DEFAULT_RECORD_TAG);
  const tagOk = /^[A-Za-z0-9_-]{1,40}$/.test(tag);

  return (
    <div className="hudbar recbar" role="toolbar" aria-label="Record demonstrations">
      {recording === null ? (
        <>
          <Button
            variant="hud"
            icon="dot"
            disabled={!ready}
            title="Record your driving as a demonstration dataset (one car, a training track, from step 0)"
            onClick={() => send({ type: "record_start" })}
            testId="btn-record"
          >
            Record
          </Button>
          {savedDataset !== null && (
            <span className="hudbar-note" data-testid="record-saved" data-dataset={savedDataset.dataset_id}>
              Saved {savedDataset.dataset_id}: {savedDataset.episodes} {savedDataset.episodes === 1 ? "drive" : "drives"},{" "}
              {savedDataset.frames} frames
            </span>
          )}
        </>
      ) : (
        <>
          <span className="rec-badge" data-testid="rec-badge" data-frames={recording.frames} data-episodes={recording.episodes}>
            ● REC {recording.episodes} {recording.episodes === 1 ? "drive" : "drives"}, {recording.frames} frames
          </span>
          <label className="hud-select" title="The dataset's tag: letters, digits, _ and -, up to 40">
            <span className="hud-select-label">Tag</span>
            <input
              type="text"
              className="hud-input"
              data-testid="record-tag"
              value={tag}
              maxLength={40}
              aria-invalid={!tagOk}
              onChange={(e) => setTag(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") e.currentTarget.blur(); // hand the arrow keys back to driving
              }}
            />
          </label>
          <Button
            variant="hud"
            icon="check"
            disabled={!tagOk || recording.frames === 0}
            title={`Save as …-keyboard-${tag}`}
            onClick={() => send({ type: "record_stop", save: true, tag })}
            testId="btn-record-save"
          >
            Save
          </Button>
          <Button variant="hud" icon="cross" title="Forget this recording" onClick={() => send({ type: "record_stop", save: false, tag })} testId="btn-record-discard">
            Discard
          </Button>
        </>
      )}
    </div>
  );
}
