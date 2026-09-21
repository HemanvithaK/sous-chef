// frontend/vite.config.js

// Vite is our build tool and dev server. Think of it as the thing
// that compiles our React code and serves it in the browser.
// We use Vite over Create React App because it's faster and simpler.

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,

    // This proxy section is critical. Here's why:
    //
    // The frontend runs on localhost:3000
    // The backend runs on localhost:8000
    //
    // Without a proxy, the browser blocks requests between
    // different ports (CORS). The proxy tells Vite:
    // "When the frontend requests /ws/*, forward it to port 8000"
    //
    // So the browser thinks it's talking to localhost:3000,
    // but Vite secretly forwards it to localhost:8000.
    proxy: {
      "/ws": {
        target: "http://localhost:8000",
        ws: true, // Enable WebSocket proxying
      },
      "/health": {
        target: "http://localhost:8000",
      },
    },
  },
});