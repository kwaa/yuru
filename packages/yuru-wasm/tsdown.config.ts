import { wasm } from 'rolldown-plugin-wasm'
import { defineConfig } from 'tsdown'

export default defineConfig({
  dts: { build: true },
  entry: ['src/index.ts', 'src/wasm-worker.js'],
  format: 'esm',
  platform: 'neutral',
  plugins: [wasm({ maxFileSize: 0, targetEnv: 'auto' })],
})
