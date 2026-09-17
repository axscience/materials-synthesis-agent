import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI backend (materials_synthesis_agent.harness.api) runs on :8000 by default.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.HARNESS_API || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
