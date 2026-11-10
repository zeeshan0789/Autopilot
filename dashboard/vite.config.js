import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  // The web API only allows this exact origin (DASHBOARD_ORIGIN), so the port must not change
  server: { port: 5173, strictPort: true },
  test: { environment: "node" },
});