import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The map base (data/reference/nyc_modzcta.geojson) is imported straight from the repo's
// reference directory as an asset, so there is exactly one copy in git and a clean clone
// still builds. `server.fs.allow` lets the dev server read one level above ui/.
export default defineConfig({
  plugins: [react()],
  assetsInclude: ["**/*.geojson"],
  server: {
    port: 5173,
    fs: { allow: [".."] },
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
