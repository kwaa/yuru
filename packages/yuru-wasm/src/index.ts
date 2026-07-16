import type { BackendCapabilities, ClothBackend, CpuIntegrationKernel } from 'yuru'

import { CPUBackend } from 'yuru'

export interface WasmBackendOptions {
  /** Injecting the module is useful in CSP-restricted deployments and tests. */
  module?: YuruWasmModule
}

export interface YuruWasmModule {
  integrate: (
    positions: Float32Array,
    previous: Float32Array,
    inverseMasses: Float32Array,
    accelerations: Float32Array,
    delta: number,
    damping: number,
  ) => Float32Array
}

class WasmIntegrationKernel implements CpuIntegrationKernel {
  readonly capabilities: BackendCapabilities

  private readonly module: YuruWasmModule

  constructor(module: YuruWasmModule) {
    this.module = module
    this.capabilities = {
      continuousCollision: true,
      kind: 'wasm-single',
      simd: true,
      threads: false,
      worker: false,
    }
  }

  integrate(
    positions: Float32Array,
    previous: Float32Array,
    inverseMasses: Float32Array,
    accelerations: Float32Array,
    delta: number,
    damping: number,
  ): void {
    const result = this.module.integrate(positions, previous, inverseMasses, accelerations, delta, damping)
    if (result.length !== positions.length * 2)
      throw new Error('yuru-wasm returned an invalid integration buffer')
    positions.set(result.subarray(0, positions.length))
    previous.set(result.subarray(positions.length))
  }
}

const importGeneratedModule = async (): Promise<YuruWasmModule> =>
  import('./generated/single/yuru_wasm_single.js')

/**
 * Creates an explicitly injectable CPU backend accelerated by Rust/WASM SIMD.
 * Rejection is intentional: callers opt into WASM and decide whether to omit
 * the backend (thereby using yuru's default TypeScript CPU implementation).
 */
export const createWasmBackend = async (options: WasmBackendOptions = {}): Promise<ClothBackend> => {
  if (typeof WebAssembly === 'undefined')
    throw new Error('WebAssembly is unavailable in this runtime')

  if (options.module != null)
    return new CPUBackend({ integrationKernel: new WasmIntegrationKernel(options.module), worker: false })

  const module = await importGeneratedModule()
  return new CPUBackend({ integrationKernel: new WasmIntegrationKernel(module), worker: false })
}
