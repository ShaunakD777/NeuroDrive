import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The prototype has its own dev server on port 5180, so it can run next to the real app (5173)
// and the Python server (8000).
//
// Two sources of data (src/source/):
//   - "Recorded": static JSON in public/recorded/, no server needed.
//   - "Live": the real Python server. Like web/, the browser talks only to this dev server, which
//     forwards /api and /ws to Python (ADR-015), so the server needs no CORS settings.
//
// `@prod` points at the production app's source (web/src/). The live adapter imports its API
// client and stores from there instead of copying them, so the prototype follows the same linking
// rules as the app (ARCHITECTURE.md §8.2). Nothing here writes to web/: it is only read.
const PYTHON_SERVER = "127.0.0.1:8000";
const PROD_SRC = fileURLToPath(new URL("../../web/src", import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@prod": PROD_SRC },
    // web/src/state/*.ts import zustand (and through it React) from web/node_modules; one copy of
    // each for the whole page, or hooks from the two copies would not work together.
    dedupe: ["react", "react-dom", "zustand"],
  },
  server: {
    host: "127.0.0.1",
    port: 5180,
    strictPort: true,
    fs: { allow: [".", PROD_SRC] },
    proxy: {
      "/api": `http://${PYTHON_SERVER}`,
      "/ws": { target: `ws://${PYTHON_SERVER}`, ws: true },
    },
  },
  preview: {
    host: "127.0.0.1",
    port: 5181,
    strictPort: true,
    proxy: {
      "/api": `http://${PYTHON_SERVER}`,
      "/ws": { target: `ws://${PYTHON_SERVER}`, ws: true },
    },
  },
});
