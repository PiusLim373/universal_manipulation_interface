import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    host: true,
    // The Python backend. In dev it runs on the host or in the container with
    // 8080 published; either way this is the only place the port is named.
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        // MJPEG and SSE are long-lived responses. Without this they are
        // buffered and the preview never paints.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => { proxyRes.headers['cache-control'] = 'no-cache' })
        },
      },
    },
  },
})
