import { describe, expect, it } from 'vitest'
import { createClothWorld } from 'yuru'

import { createWasmBackend } from '../src/index'

import * as wasmModule from '../src/generated/single/yuru_wasm_single.js'

describe('wasm backend', () => {
  it('loads the generated no_std module through the default factory path', async () => {
    const backend = await createWasmBackend()
    expect(backend.capabilities).toMatchObject({
      kind: 'wasm-single',
      simd: true,
      threads: false,
      worker: false,
    })
    backend.dispose()
  })

  it('runs the generated SIMD integration kernel in a cloth world', async () => {
    const backend = await createWasmBackend({ module: wasmModule })
    const world = createClothWorld({
      backend,
      fixedDelta: 1 / 60,
      gravity: [0, -9.81, 0],
      quality: { collisionEverySubsteps: 1, substeps: 1 },
    })
    const body = world.addBody({
      mesh: {
        indices: new Uint16Array([0, 1, 2]),
        inverseMasses: new Float32Array([0, 1, 1]),
        positions: new Float32Array([0, 1, 0, 1, 1, 0, 0, 0, 0]),
      },
      selfCollision: false,
    })

    await world.step(1 / 60)

    expect(world.diagnostics.backend).toMatchObject({ kind: 'wasm-single', simd: true })
    expect(world.getPositions(body)[4]).toBeLessThan(1)
    world.dispose()
  })
})
