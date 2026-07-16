import { defineConfig } from '@moeru/eslint-config'

export default defineConfig({
  react: { reactCompiler: true },
})
  .append({
    ignores: [
      'app/src/router.ts',
      'packages/yuru-wasm/src/generated/**',
    ],
  })
