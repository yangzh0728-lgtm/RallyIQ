import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Dev: /api is proxied to the FastAPI backend (uvicorn rallyiq.api.app:app --port 8000).
// Set VITE_USE_MOCK=1 to run fully offline against generated mock data.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: process.env.RALLYIQ_API ?? "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  test: { environment: "node" },
});
