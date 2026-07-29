import { afterEach, describe, expect, it, vi } from 'vitest'
import { CPUBackend, createClothWorld } from 'yuru'

import { createWasmBackend } from '../src/index'

import * as wasmModule from '../src/generated/single/yuru_wasm_single.js'

describe('wasm backend', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('uses a dedicated module Worker in browsers', async () => {
    const workers: Array<{ messages: unknown[], options: WorkerOptions, url: URL }> = []
    class MockWorker {
      readonly messages: unknown[] = []
      readonly options: WorkerOptions
      readonly url: URL

      constructor(url: URL, options: WorkerOptions) {
        this.options = options
        this.url = url
        workers.push(this)
      }

      addEventListener(type: string, listener: (event: { data: unknown }) => void): void {
        if (type !== 'message')
          return
        queueMicrotask(() => listener({ data: { type: 'ready' } }))
      }

      postMessage(message: unknown): void {
        this.messages.push(message)
      }

      terminate(): void {}
    }
    vi.stubGlobal('Worker', MockWorker)

    const backend = await createWasmBackend()
    backend.addBody({
      mesh: {
        indices: new Uint16Array([0, 1, 2]),
        positions: new Float32Array(9),
      },
      selfCollision: false,
    })

    expect(backend.capabilities).toMatchObject({
      kind: 'wasm-single',
      simd: true,
      worker: true,
    })
    expect(workers).toHaveLength(1)
    expect(workers[0].options).toMatchObject({ name: 'yuru-wasm', type: 'module' })
    expect(workers[0].url.pathname).toMatch(/wasm-worker\.js$/)
    expect(workers[0].messages[0]).toMatchObject({ type: 'addBody' })
    backend.dispose()
  })

  it('waits for the WASM Worker ready handshake before resolving', async () => {
    let messageListener: ((event: { data: unknown }) => void) | undefined
    class MockWorker {
      constructor(_url: URL, _options: WorkerOptions) {}

      addEventListener(type: string, listener: (event: { data: unknown }) => void): void {
        if (type !== 'message')
          return
        messageListener = listener
      }

      postMessage(): void {}

      terminate(): void {}
    }
    vi.stubGlobal('Worker', MockWorker)

    let settled = false
    const backendPromise = createWasmBackend()
    void backendPromise.then(() => {
      settled = true
    })
    await Promise.resolve()
    expect(settled).toBe(false)
    messageListener?.({ data: { type: 'ready' } })
    const backend = await backendPromise
    expect(settled).toBe(true)
    backend.dispose()
  })

  it('rejects when the WASM Worker fails before ready', async () => {
    let errorListener: ((event: { message?: string }) => void) | undefined
    class MockWorker {
      constructor(_url: URL, _options: WorkerOptions) {}

      addEventListener(type: string, _listener: (event: { data: unknown }) => void): void {
        if (type !== 'error')
          return
        errorListener = _listener as unknown as (event: { message?: string }) => void
      }

      postMessage(): void {}

      terminate(): void {}
    }
    vi.stubGlobal('Worker', MockWorker)

    const backendPromise = createWasmBackend()
    errorListener?.({ message: 'WASM module failed to load' })
    await expect(backendPromise).rejects.toThrow('WASM module failed to load')
  })

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

  it('matches CPU wind ordering before speed and structural constraints', async () => {
    const mesh = {
      indices: new Uint16Array([0, 1, 2]),
      inverseMasses: new Float32Array([1, 1, 1]),
      positions: new Float32Array([0, 1, 0, 1, 1, 0, 0, 0, 0]),
    }
    const options = {
      fixedDelta: 1 / 60,
      forceFields: [{ type: 'wind' as const, velocity: [0, 0, 50] as const }],
      gravity: [0, 0, 0] as const,
      quality: { collisionEverySubsteps: 1, collisionIterations: 0, substeps: 1 },
      speedLimit: 0.1,
    }
    const descriptor = {
      materials: [{ bendCompliance: 1e6, drag: 0.5, lift: 0.1, shearCompliance: 1e6, stretchCompliance: 1e6 }],
      mesh,
      selfCollision: false,
    }
    const cpuWorld = createClothWorld({ ...options, backend: new CPUBackend({ worker: false }) })
    const wasmWorld = createClothWorld({
      ...options,
      backend: await createWasmBackend({ module: wasmModule }),
    })
    const cpuBody = cpuWorld.addBody(descriptor)
    const wasmBody = wasmWorld.addBody(descriptor)

    await cpuWorld.step(1 / 60)
    await wasmWorld.step(1 / 60)

    const cpuPositions = cpuWorld.getPositions(cpuBody)
    const wasmPositions = wasmWorld.getPositions(wasmBody)
    for (let index = 0; index < cpuPositions.length; index++)
      expect(wasmPositions[index]).toBeCloseTo(cpuPositions[index], 5)
    expect(wasmPositions[2]).toBeGreaterThan(0)
    cpuWorld.dispose()
    wasmWorld.dispose()
  })
})
