import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The map base is not bundled: NeedMap asks `GET /region` for the region's geojson, or reads
// the copy scripts/make_ui_fixtures.py puts beside the fixtures for a build with no API.
export default defineConfig({
  // Relative, so the built bundle works served from any sub-path, not just a domain root.
  base: "./",
  plugins: [react()],
  server: {
    port: 5173,
    // The UI calls /api/*; in dev that is proxied to the FastAPI app on :8000. When the
    // API is not running the proxy fails fast and the client falls back to fixtures.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
  build: { chunkSizeWarningLimit: 1500 },
});
