import type { HelloMessage } from "./generated";

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

const RETRY_MS = 2000;

/** The live socket address, on the same origin as the page (the dev server forwards /ws). */
function liveUrl(): string {
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${window.location.host}/ws/live`;
}

function isHello(value: unknown): value is HelloMessage {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return v.type === "hello" && typeof v.version === "string" && typeof v.api === "string";
}

/**
 * Keep a WebSocket open to the Python server and report its status.
 *
 * The status becomes "connected" only after the server's `hello` message arrives, so it proves
 * the server speaks the same message shapes as the generated types. If the socket closes or
 * cannot be opened, the status becomes "disconnected" and a new attempt starts every 2 seconds.
 * Returns a function that stops everything.
 */
export function connectLive(onStatus: (status: ConnectionStatus) => void): () => void {
  let socket: WebSocket | null = null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let stopped = false;

  function open(): void {
    onStatus("connecting");
    const ws = new WebSocket(liveUrl());
    socket = ws;

    ws.onmessage = (event: MessageEvent) => {
      try {
        if (isHello(JSON.parse(String(event.data)))) onStatus("connected");
      } catch {
        // Not JSON: ignore. Only the hello message matters for the status.
      }
    };
    ws.onclose = () => {
      if (stopped) return;
      onStatus("disconnected");
      timer = setTimeout(open, RETRY_MS);
    };
  }

  open();

  return () => {
    stopped = true;
    clearTimeout(timer);
    if (socket === null) return;
    socket.onmessage = null;
    socket.onclose = null;
    if (socket.readyState === WebSocket.OPEN) socket.close();
    // Closing a socket that is still connecting would log a browser warning; close it on open.
    else if (socket.readyState === WebSocket.CONNECTING) {
      const pending = socket;
      pending.onopen = () => pending.close();
    }
  };
}
