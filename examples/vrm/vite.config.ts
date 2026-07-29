import react from '@vitejs/plugin-react'

import { defineConfig } from 'vite'

const crossOriginIsolationHeaders = {
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Opener-Policy': 'same-origin',
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  preview: { headers: crossOriginIsolationHeaders },
  resolve: { dedupe: ['react', 'three'] },
  server: { headers: crossOriginIsolationHeaders },
  worker: { format: 'es' },
})
