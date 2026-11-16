import { useEffect, useRef, useState } from "react";

import { connectLive, type ConnectionStatus, type LiveConnection } from "../api/ws";
import { applyRunsMessage, historyFor, initialRunsWatch, type RunsWatchState, watchRun } from "./runsWatch";

/**
 * The `/ws/runs` socket (task 5.8, ADR-041) for one tool on screen: every run's status, and the
 * history of the run `watched` (its metrics lines, checkpoints and exam headlines). The socket is
 * opened once and closed when the tool goes; after a reconnect the server knows nothing of this
 * page, so `watch` is sent again. When `watched` changes, the old history is dropped and the new
 * run's is asked for. Until then `historyFor` hands out an empty history for the new run, never
 * the previous run's (task S0; this replaces the copies in Timeline.tsx and TrainingRuns.tsx).
 */
export function useRunsSocket(watched: string | null): { watch: RunsWatchState; status: ConnectionStatus } {
  const [watch, setWatch] = useState<RunsWatchState>(initialRunsWatch);
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const socket = useRef<LiveConnection | null>(null);
  const current = useRef<string | null>(watched);
  current.current = watched;

  useEffect(() => {
    const connection = connectLive(
      (s) => {
        setStatus(s);
        if (s === "connected" && current.current !== null)
          socket.current?.send(JSON.stringify({ type: "watch", run_id: current.current }));
      },
      (message) => setWatch((state) => applyRunsMessage(state, message)),
      "/ws/runs",
    );
    socket.current = connection;
    return () => {
      socket.current = null;
      connection.stop();
    };
  }, []);

  useEffect(() => {
    setWatch((state) => watchRun(state, watched));
    socket.current?.send(JSON.stringify({ type: "watch", run_id: watched }));
  }, [watched]);

  return { watch: historyFor(watch, watched), status };
}
