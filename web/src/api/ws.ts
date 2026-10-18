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

/** What `connectLive` hands back: a way to send text and a way to stop everything. */
export interface LiveConnection {
  /** Send one text message; returns false (and sends nothing) if the socket is not open. */
  send: (text: string) => boolean;
  stop: () => void;
}

/**
 * Keep a WebSocket open to the Python server and report its status.
 *
 * The status becomes "connected" only after the server's `hello` message arrives, so it proves
 * the server speaks the same message shapes as the generated types. Every message that parses
 * as JSON (the hello included) is also passed to `onMessage`. If the socket closes or cannot be
 * opened, the status becomes "disconnected" and a new attempt starts every 2 seconds.
 */
export function connectLive(
  onStatus: (status: ConnectionStatus) => void,
  onMessage?: (message: unknown) => void,
): LiveConnection {
  let socket: WebSocket | null = null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let stopped = false;

  function open(): void {
    onStatus("connecting");
    const ws = new WebSocket(liveUrl());
    socket = ws;

    ws.onmessage = (event: MessageEvent) => {
      let message: unknown;
      try {
        message = JSON.parse(String(event.data));
      } catch {
        return; // Not JSON: ignore.
      }
      if (isHello(message)) onStatus("connected");
      onMessage?.(message);
    };
    ws.onclose = () => {
      if (stopped) return;
      onStatus("disconnected");
      timer = setTimeout(open, RETRY_MS);
    };
  }

  open();

  const stop = () => {
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
  const send = (text: string): boolean => {
    if (socket === null || socket.readyState !== WebSocket.OPEN) return false;
    socket.send(text);
    return true;
  };

  return { send, stop };
}
