import type {
  BackendCapabilities,
  BackendStepOptions,
  BodyId,
  ClothBackend,
  ClothBodyDescriptor,
  ColliderDescriptor,
  ColliderId,
  CpuIntegrationKernel,
  GrabDescriptor,
  GrabId,
  Vec3Like,
} from './types.js'

import { CPUSolverBackend } from './cpu-solver-backend.js'
import { readVec3 } from './math.js'

export interface CPUBackendOptions {
  /** Custom numeric kernels stay on the calling thread because they are not structured-cloneable. */
  integrationKernel?: CpuIntegrationKernel
  /** Defaults to a dedicated module Worker when the runtime supports it. */
  worker?: boolean | CPUWorkerLike
}

export interface CPUWorkerLike {
  onerror: ((event: { message?: string }) => void) | null
  onmessage: ((event: { data: unknown }) => void) | null
  postMessage: (message: unknown) => void
  terminate: () => void
}

interface BodyMirror {
  initial: Float32Array
  positions: Float32Array
}

interface ErrorResponse {
  message: string
  requestId?: number
  stack?: string
  type: 'error'
}

interface StepResponse {
  positions: [BodyId, Float32Array][]
  requestId: number
  type: 'step'
}

type WorkerResponse = ErrorResponse | StepResponse

const workerCapabilities: BackendCapabilities = {
  continuousCollision: true,
  kind: 'cpu',
  simd: false,
  threads: false,
  worker: true,
}

// eslint-disable-next-line sonarjs/function-return-type -- Feature detection intentionally returns a Worker or no value.
const createDefaultWorker = (): CPUWorkerLike | undefined => {
  if (typeof Worker === 'undefined')
    return undefined
  const worker = new Worker(new URL('./cpu-worker.js', import.meta.url), {
    name: 'yuru-cpu',
    type: 'module',
  })
  const adapter: CPUWorkerLike = {
    onerror: null,
    onmessage: null,
    postMessage: message => worker.postMessage(message),
    terminate: () => worker.terminate(),
  }
  worker.addEventListener('message', (event) => {
    adapter.onmessage?.({ data: event.data as WorkerResponse })
  })
  worker.addEventListener('error', (event) => {
    adapter.onerror?.({ message: event.message })
  })
  return adapter
}

const validateBody = (descriptor: ClothBodyDescriptor): void => {
  const { mesh } = descriptor
  if (mesh.positions.length === 0 || mesh.positions.length % 3 !== 0)
    throw new RangeError('Cloth positions must contain packed xyz values')
  if (mesh.indices.length === 0 || mesh.indices.length % 3 !== 0)
    throw new RangeError('Cloth indices must contain triangles')
  const count = mesh.positions.length / 3
  if (mesh.inverseMasses != null && mesh.inverseMasses.length !== count)
    throw new RangeError('inverseMasses must contain one value per particle')
  if (mesh.triangleMaterialIndices != null && mesh.triangleMaterialIndices.length !== mesh.indices.length / 3)
    throw new RangeError('triangleMaterialIndices must contain one value per triangle')
}

/**
 * TypeScript CPU backend. Its default browser path is a dedicated Worker;
 * runtimes without Worker support use the same typed-array solver inline.
 */
export class CPUBackend implements ClothBackend {
  readonly capabilities: BackendCapabilities

  private readonly bodies = new Map<BodyId, BodyMirror>()
  private readonly direct?: CPUSolverBackend
  private disposed = false
  private fatalError?: Error
  private nextBodyId = 1
  private nextColliderId = 1
  private nextGrabId = 1
  private nextRequestId = 1
  private readonly pending = new Map<number, { reject: (error: Error) => void, resolve: () => void }>()
  private readonly worker?: CPUWorkerLike

  constructor(options: CPUBackendOptions = {}) {
    const requestedWorker = options.integrationKernel == null && options.worker !== false
    const worker = typeof options.worker === 'object'
      ? options.worker
      : requestedWorker
        ? createDefaultWorker()
        : undefined
    if (worker == null) {
      this.direct = new CPUSolverBackend(options.integrationKernel)
      this.capabilities = this.direct.capabilities
      return
    }
    this.worker = worker
    this.capabilities = workerCapabilities
    worker.onmessage = event => this.handleMessage(event.data as WorkerResponse)
    worker.onerror = (event) => {
      this.fail(new Error(event.message ?? 'Yuru CPU Worker failed'))
    }
  }

  addBody(descriptor: ClothBodyDescriptor): BodyId {
    if (this.direct != null)
      return this.direct.addBody(descriptor)
    this.assertActive()
    validateBody(descriptor)
    const id = this.nextBodyId++
    const positions = descriptor.mesh.positions.slice()
    this.bodies.set(id, { initial: positions.slice(), positions })
    this.post({ descriptor, id, type: 'addBody' })
    return id
  }

  addCollider(descriptor: ColliderDescriptor): ColliderId {
    if (this.direct != null)
      return this.direct.addCollider(descriptor)
    this.assertActive()
    const id = this.nextColliderId++
    this.post({ descriptor, id, type: 'addCollider' })
    return id
  }

  addGrab(descriptor: GrabDescriptor): GrabId {
    if (this.direct != null)
      return this.direct.addGrab(descriptor)
    this.assertActive()
    this.requireBody(descriptor.body)
    const id = this.nextGrabId++
    this.post({ descriptor, id, type: 'addGrab' })
    return id
  }

  dispose(): void {
    if (this.disposed)
      return
    this.disposed = true
    this.direct?.dispose()
    this.worker?.terminate()
    this.bodies.clear()
    this.fail(new Error('CPUBackend has been disposed'))
  }

  getPositions(id: BodyId): Float32Array {
    return this.direct?.getPositions(id) ?? this.requireBody(id).positions
  }

  removeBody(id: BodyId): void {
    if (this.direct != null) {
      this.direct.removeBody(id)
      return
    }
    this.assertActive()
    this.requireBody(id)
    this.bodies.delete(id)
    this.post({ id, type: 'removeBody' })
  }

  removeCollider(id: ColliderId): void {
    if (this.direct != null) {
      this.direct.removeCollider(id)
      return
    }
    this.assertActive()
    this.post({ id, type: 'removeCollider' })
  }

  removeGrab(id: GrabId): void {
    if (this.direct != null) {
      this.direct.removeGrab(id)
      return
    }
    this.assertActive()
    this.post({ id, type: 'removeGrab' })
  }

  resetBody(id: BodyId, positions?: Float32Array): void {
    if (this.direct != null) {
      this.direct.resetBody(id, positions)
      return
    }
    this.assertActive()
    const body = this.requireBody(id)
    const source = positions ?? body.initial
    if (source.length !== body.positions.length)
      throw new RangeError('Reset positions do not match the body particle count')
    body.positions.set(source)
    this.post({ id, positions, type: 'resetBody' })
  }

  setParticleTargets(id: BodyId, indices: Uint32Array, positions: Float32Array): void {
    if (this.direct != null) {
      this.direct.setParticleTargets(id, indices, positions)
      return
    }
    this.assertActive()
    const body = this.requireBody(id)
    if (positions.length !== indices.length * 3)
      throw new RangeError('Particle target positions must match target indices')
    for (const index of indices) {
      if (index * 3 >= body.positions.length)
        throw new RangeError(`Particle target ${index} is outside body ${id}`)
    }
    this.post({ id, indices, positions, type: 'setParticleTargets' })
  }

  step(delta: number, options: BackendStepOptions): Promise<void> | void {
    if (this.direct != null)
      return this.direct.step(delta, options)
    this.assertActive()
    const requestId = this.nextRequestId++
    return new Promise<void>((resolve, reject) => {
      this.pending.set(requestId, { reject, resolve })
      this.post({ delta, options, requestId, type: 'step' })
    })
  }

  updateCollider(id: ColliderId, descriptor: ColliderDescriptor): void {
    if (this.direct != null) {
      this.direct.updateCollider(id, descriptor)
      return
    }
    this.assertActive()
    this.post({ descriptor, id, type: 'updateCollider' })
  }

  updateGrab(id: GrabId, position: Vec3Like): void {
    if (this.direct != null) {
      this.direct.updateGrab(id, position)
      return
    }
    this.assertActive()
    this.post({ id, position: readVec3(position), type: 'updateGrab' })
  }

  private assertActive(): void {
    if (this.disposed)
      throw new Error('CPUBackend has been disposed')
    if (this.fatalError != null)
      throw this.fatalError
  }

  private fail(error: Error): void {
    this.fatalError = error
    for (const pending of this.pending.values())
      pending.reject(error)
    this.pending.clear()
  }

  private handleMessage(message: WorkerResponse): void {
    if (message.type === 'error') {
      const error = new Error(message.message)
      if (message.stack != null)
        error.stack = message.stack
      if (message.requestId == null) {
        this.fail(error)
        return
      }
      this.pending.get(message.requestId)?.reject(error)
      this.pending.delete(message.requestId)
      return
    }
    for (const [id, positions] of message.positions)
      this.bodies.get(id)?.positions.set(positions)
    this.pending.get(message.requestId)?.resolve()
    this.pending.delete(message.requestId)
  }

  private post(message: unknown): void {
    this.worker?.postMessage(message)
  }

  private requireBody(id: BodyId): BodyMirror {
    const body = this.bodies.get(id)
    if (body == null)
      throw new RangeError(`Unknown cloth body ${id}`)
    return body
  }
}
