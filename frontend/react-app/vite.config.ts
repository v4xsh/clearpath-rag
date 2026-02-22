import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendUrl = env.VITE_API_URL || "http://localhost:8000";

  return {
    plugins: [react()],

    server: {
      port: 5173,
      strictPort: true,
      proxy: {
        "/api": {
          target: backendUrl,
          changeOrigin: true,
          rewrite: (path) => path,
        },
      },
    },

    build: {
      outDir: "dist",
      sourcemap: true,
      target: "es2020",
    },

    preview: {
      port: 4173,
    },
  };
});