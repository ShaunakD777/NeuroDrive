import { useEffect } from "react";

import { attachSender, useLive } from "../state/live";
import { fetchTrack, fetchTracks } from "./rest";
import { connectLive } from "./ws";

/**
 * Run the app's single live connection and keep the live store filled.
 *
 * 1. Open the WebSocket; every message goes into the store, and so does the connection status.
 * 2. Each time the status becomes "connected" (first connect, or after the server restarted),
 *    fetch the track list and load a scenario: the track the user chose, or the first one.
 * 3. Whenever the loaded scenario names a track whose outline is not in the store, fetch it.
 */
export function useLiveConnection(): void {
  const status = useLive((s) => s.status);
  const scenarioTrack = useLive((s) => s.scenario?.track ?? null);
  const geometryName = useLive((s) => s.geometry?.name ?? null);

  useEffect(() => {
    const { setStatus, receive } = useLive.getState();
    const connection = connectLive(setStatus, receive);
    attachSender(connection.send);
    return () => {
      attachSender(null);
      connection.stop();
    };
  }, []);

  useEffect(() => {
    if (status !== "connected") return;
    let cancelled = false;
    void (async () => {
      try {
        const { tracks } = await fetchTracks();
        if (cancelled) return;
        const live = useLive.getState();
        live.setTracks(tracks);
        const name = live.wantedTrack ?? tracks[0]?.name;
        if (name !== undefined) live.loadTrack(name);
      } catch {
        // The server could not list tracks; the view stays empty until the next reconnect.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [status]);

  useEffect(() => {
    if (scenarioTrack === null || scenarioTrack === geometryName) return;
    let cancelled = false;
    fetchTrack(scenarioTrack)
      .then((geometry) => {
        if (!cancelled) useLive.getState().setGeometry(geometry);
      })
      .catch(() => {
        // Outline unavailable: the cars are still drawn, without a road.
      });
    return () => {
      cancelled = true;
    };
  }, [scenarioTrack, geometryName]);
}
