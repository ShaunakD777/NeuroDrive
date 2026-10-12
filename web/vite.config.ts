import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The Python server (`nd serve`) listens on 127.0.0.1:8000. In development the browser talks only
// to this dev server (port 5173), which forwards /api and /ws to Python. The browser therefore
// sees a single origin, so the Python server needs no CORS settings (ADR-015).
const PYTHON_SERVER = "127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": `http://${PYTHON_SERVER}`,
      "/ws": { target: `ws://${PYTHON_SERVER}`, ws: true },
    },
  },
  test: {
    include: ["src/**/*.test.ts"],
  },
});
