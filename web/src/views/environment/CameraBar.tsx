import { Button, Segmented } from "../../components/ui/controls";
import type { CameraMode } from "./EnvironmentView";

// Follow the car, the whole track, the free-view note and zoom, floating at the top right of the
// stage (task E3, recommendation E12). The mode lives in EnvironmentView, which owns the camera.

export function CameraBar({
  mode,
  onModeChange,
  onZoomIn,
  onZoomOut,
}: {
  mode: CameraMode;
  onModeChange: (mode: "follow" | "fit") => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
}) {
  return (
    <div className="hudbar" role="toolbar" aria-label="Camera">
      <Segmented
        variant="hud"
        label="Camera"
        value={mode === "free" ? ("free" as const) : mode}
        onChange={(id) => id !== "free" && onModeChange(id)}
        options={[
          { id: "follow", label: "Follow car", icon: "follow", hint: "Follow the selected car (or double-click the track)", testId: "camera-follow" },
          { id: "fit", label: "Whole track", icon: "fit", hint: "Show the whole track", testId: "camera-fit" },
        ]}
      />
      {mode === "free" && (
        <span className="hudbar-note" data-testid="camera-free" title="You moved the view: it stays put until Follow car, Whole track or a double-click">
          Free view
        </span>
      )}
      <Button variant="hud" icon="minus" title="Zoom out (or scroll over the track)" onClick={onZoomOut} testId="zoom-out" />
      <Button variant="hud" icon="plus" title="Zoom in (or scroll over the track)" onClick={onZoomIn} testId="zoom-in" />
    </div>
  );
}
