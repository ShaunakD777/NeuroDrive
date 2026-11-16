import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The prototype has its own dev server on port 5180, so it can run next to the real app
// (5173) and the Python server (8000). It never proxies to Python: all data is static JSON
// in public/data/.
export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", port: 5180, strictPort: true },
  preview: { host: "127.0.0.1", port: 5181, strictPort: true },
});
