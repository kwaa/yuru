import type {
  BackendCapabilities,
  BodyId,
  ClothBackend,
  CpuIntegrationKernel,
  CpuSolverBodyState,
  CPUWorkerLike,
} from 'yuru'

import { CPUBackend } from 'yuru'

export interface WasmBackendOptions {
  /** Injecting the module is useful in CSP-restricted deployments and tests. */
  module?: YuruWasmModule
  /** Defaults to a dedicated WASM Worker when the runtime supports it. */
  worker?: boolean | CPUWorkerLike
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
  WasmSolver?: new () => YuruWasmSolver
}

export interface YuruWasmSolver {
  add_body: (
    id: number,
    initial: Float32Array,
    inverseMasses: Float32Array,
    indices: Uint32Array,
    adjacency: Uint32Array,
    adjacencyOffsets: Uint32Array,
    edgeKinds: Uint8Array,
    edgeParticles: Uint32Array,
    edgeRestLengths: Float32Array,
    triangleParticles: Uint32Array,
    triangleRestAreas: Float32Array,
    tetherAnchors: Uint32Array,
    tetherLengths: Float32Array,
    tetherParticles: Uint32Array,
    materialThicknesses: Float32Array,
    materialFrictions: Float32Array,
    materialDrags: Float32Array,
    materialLifts: Float32Array,
    triangleMaterialIndices: Uint16Array,
    collisionCellSize: number,
    collisionLayer: number,
    collisionLayerAxis: Float32Array,
    filterGroup: number,
    filterMask: number,
    selfCollision: boolean,
  ) => void
  body_state: (id: number) => Float32Array
  free?: () => void
  has_body: (id: number) => boolean
  remove_body: (id: number) => void
  solve_body_collisions: (maxCandidates: number) => void
  solve_edge_collisions: (maxCandidates: number) => void
  solve_structural: (
    id: number,
    positions: Float32Array,
    previous: Float32Array,
    accelerations: Float32Array,
    delta: number,
    damping: number,
    maximumDisplacement: number,
    maximumDepenetration: number,
    stretchCompliance: number,
    bendCompliance: number,
    shearCompliance: number,
    windX: number,
    windY: number,
    windZ: number,
    hasWind: boolean,
  ) => Float32Array
  sync_body: (
    id: number,
    positions: Float32Array,
    previous: Float32Array,
    maximumDepenetration: number,
  ) => void
}

export class WasmIntegrationKernel implements CpuIntegrationKernel {
  readonly capabilities: BackendCapabilities
  readonly integratesAerodynamics: boolean
  readonly integratesStructuralConstraints: boolean
  readonly solveBodyCollisions?: (
    bodies: readonly CpuSolverBodyState[],
    maxCandidates: number,
  ) => void

  readonly solveEdgeCollisions?: (
    bodies: readonly CpuSolverBodyState[],
    maxCandidates: number,
  ) => void

  private readonly module: YuruWasmModule
  private readonly solver?: YuruWasmSolver

  constructor(module: YuruWasmModule) {
    this.module = module
    this.solver = module.WasmSolver == null ? undefined : new module.WasmSolver()
    this.integratesStructuralConstraints = this.solver != null
    this.integratesAerodynamics = this.solver != null
    if (this.solver != null) {
      this.solveBodyCollisions = (bodies, maxCandidates) => this.solveCollisions(bodies, maxCandidates)
      // solveCollisions batches vertex/triangle and edge/edge phases through
      // one state sync. The CPU solver calls this immediately afterward.
      this.solveEdgeCollisions = () => {}
    }
    this.capabilities = {
      continuousCollision: true,
      kind: 'wasm-single',
      simd: true,
      threads: false,
      worker: false,
    }
  }

  dispose(): void {
    this.solver?.free?.()
  }

  integrate(
    positions: Float32Array,
    previous: Float32Array,
    inverseMasses: Float32Array,
    accelerations: Float32Array,
    delta: number,
    damping: number,
    body?: CpuSolverBodyState,
    wind?: readonly [number, number, number],
  ): void {
    const result = this.solver == null || body == null
      ? this.module.integrate(positions, previous, inverseMasses, accelerations, delta, damping)
      : this.solveStructural(body, positions, previous, accelerations, delta, damping, wind)
    if (result.length !== positions.length * 2)
      throw new Error('yuru-wasm returned an invalid integration buffer')
    positions.set(result.subarray(0, positions.length))
    previous.set(result.subarray(positions.length))
  }

  removeBody(id: BodyId): void {
    this.solver?.remove_body(id)
  }

  private registerBody(body: CpuSolverBodyState): void {
    if (this.solver == null || this.solver.has_body(body.id))
      return
    const tethers = body.tethers?.topology
    this.solver.add_body(
      body.id,
      body.initial,
      body.inverseMasses,
      body.indices instanceof Uint32Array ? body.indices : Uint32Array.from(body.indices),
      body.topology.adjacency,
      body.topology.adjacencyOffsets,
      body.topology.edgeKinds,
      body.topology.edgeParticles,
      body.topology.edgeRestLengths,
      body.topology.triangleParticles,
      body.topology.triangleRestAreas,
      tethers?.anchors ?? new Uint32Array(),
      tethers?.lengths ?? new Float32Array(),
      tethers?.particles ?? new Uint32Array(),
      Float32Array.from(body.materials, material => material.thickness),
      Float32Array.from(body.materials, material => material.kineticFriction),
      Float32Array.from(body.materials, material => material.drag),
      Float32Array.from(body.materials, material => material.lift),
      body.triangleMaterialIndices ?? new Uint16Array(),
      body.collisionCellSize,
      body.collisionLayer,
      body.collisionLayerAxis == null ? new Float32Array() : Float32Array.from(body.collisionLayerAxis),
      body.filter.group,
      body.filter.mask,
      body.selfCollision,
    )
  }

  private solveCollisions(bodies: readonly CpuSolverBodyState[], maxCandidates: number): void {
    const solver = this.solver!
    for (const body of bodies) {
      this.registerBody(body)
      solver.sync_body(
        body.id,
        body.positions,
        body.previous,
        body.maximumSelfCollisionDepenetration,
      )
    }
    solver.solve_body_collisions(maxCandidates)
    solver.solve_edge_collisions(maxCandidates)
    for (const body of bodies) {
      const state = solver.body_state(body.id)
      if (state.length !== body.positions.length * 2)
        throw new Error('yuru-wasm returned an invalid collision buffer')
      body.positions.set(state.subarray(0, body.positions.length))
      body.previous.set(state.subarray(body.positions.length))
    }
  }

  private solveStructural(
    body: CpuSolverBodyState,
    positions: Float32Array,
    previous: Float32Array,
    accelerations: Float32Array,
    delta: number,
    damping: number,
    wind?: readonly [number, number, number],
  ): Float32Array {
    this.registerBody(body)
    return this.solver!.solve_structural(
      body.id,
      positions,
      previous,
      accelerations,
      delta,
      damping,
      body.maximumSelfCollisionDisplacement,
      body.maximumSelfCollisionDepenetration,
      body.materials[0].stretchCompliance,
      body.materials[0].bendCompliance,
      body.materials[0].shearCompliance,
      wind?.[0] ?? 0,
      wind?.[1] ?? 0,
      wind?.[2] ?? 0,
      wind != null,
    )
  }
}

const importGeneratedModule = async (): Promise<YuruWasmModule> =>
  import('./generated/single/yuru_wasm_single.js')

const wasmWorkerCapabilities: BackendCapabilities = {
  continuousCollision: true,
  kind: 'wasm-single',
  simd: true,
  threads: false,
  worker: true,
}

const createDefaultWasmWorker = async (): Promise<CPUWorkerLike | undefined> => {
  if (typeof Worker === 'undefined')
    return undefined
  const worker = new Worker(new URL('./wasm-worker.js', import.meta.url), {
    name: 'yuru-wasm',
    type: 'module',
  })
  const adapter: CPUWorkerLike = {
    onerror: null,
    onmessage: null,
    postMessage: message => worker.postMessage(message),
    terminate: () => worker.terminate(),
  }
  let ready = false
  await new Promise<void>((resolve, reject) => {
    worker.addEventListener('message', (event) => {
      const data = event.data as { message?: string, stack?: string, type?: string }
      if (!ready) {
        if (data.type === 'ready') {
          ready = true
          resolve()
          return
        }
        const error = new Error(data.message ?? 'Yuru WASM Worker sent a message before initialization')
        if (data.stack != null)
          error.stack = data.stack
        worker.terminate()
        reject(error)
        return
      }
      adapter.onmessage?.({ data: event.data as unknown })
    })
    worker.addEventListener('error', (event) => {
      const error = new Error(event.message || 'Yuru WASM Worker failed to initialize')
      if (!ready) {
        worker.terminate()
        reject(error)
        return
      }
      adapter.onerror?.({ message: error.message })
    })
  })
  return adapter
}

/**
 * Creates a Rust/WASM SIMD backend. Browsers use a dedicated Worker by default
 * and share positions through SAB/Atomics when cross-origin isolation allows
 * it. Runtimes without Worker support execute the same kernel inline.
 */
export const createWasmBackend = async (options: WasmBackendOptions = {}): Promise<ClothBackend> => {
  if (typeof WebAssembly === 'undefined')
    throw new Error('WebAssembly is unavailable in this runtime')

  if (options.module != null)
    return new CPUBackend({ integrationKernel: new WasmIntegrationKernel(options.module), worker: false })

  const worker = typeof options.worker === 'object'
    ? options.worker
    : options.worker === false
      ? undefined
      : await createDefaultWasmWorker()
  if (worker != null) {
    return new CPUBackend({
      worker,
      workerCapabilities: wasmWorkerCapabilities,
    })
  }

  const module = await importGeneratedModule()
  return new CPUBackend({ integrationKernel: new WasmIntegrationKernel(module), worker: false })
}
