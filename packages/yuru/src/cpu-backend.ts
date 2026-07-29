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
  /** Identifies a custom Worker implementation such as the WASM backend. */
  workerCapabilities?: BackendCapabilities
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
  sharedWithWorker: boolean
}

interface ErrorResponse {
  message: string
  requestId?: number
  stack?: string
  type: 'error'
}

interface PendingStep {
  reject: (error: Error) => void
  resolve: () => void
  signal?: Int32Array
}

interface StepResponse {
  positions: [BodyId, Float32Array][]
  requestId: number
  type: 'step'
}

interface WorkerReadyResponse {
  type: 'ready'
}

type WorkerResponse = ErrorResponse | StepResponse | WorkerReadyResponse

const supportsSharedTransport = (): boolean =>
  typeof SharedArrayBuffer !== 'undefined'
  && typeof Atomics !== 'undefined'
  && typeof Atomics.waitAsync === 'function'

const createPositions = (source: Float32Array, shared: boolean): Float32Array => {
  const positions = shared
    ? new Float32Array(new SharedArrayBuffer(source.byteLength))
    : new Float32Array(source.length)
  positions.set(source)
  return positions
}

const isSharedTransportError = (error: unknown): boolean => {
  if (typeof error !== 'object' || error == null || !('name' in error))
    return false
  return error.name === 'DataCloneError' || error.name === 'SecurityError'
}

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
  if (descriptor.motionConstraints !== false && descriptor.motionConstraints != null) {
    if (descriptor.motionConstraints.maximumDistances.length !== count)
      throw new RangeError('Motion maximumDistances must contain one value per particle')
    if (descriptor.motionConstraints.targets != null && descriptor.motionConstraints.targets.length !== mesh.positions.length)
      throw new RangeError('Motion targets must contain one packed xyz value per particle')
  }
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
  private readonly pending = new Map<number, PendingStep>()
  private sharedTransport = false
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
    this.capabilities = options.workerCapabilities ?? workerCapabilities
    if (supportsSharedTransport()) {
      try {
        // Construction is a stricter capability check than the global alone in
        // runtimes that gate shared memory behind cross-origin isolation.
        this.sharedTransport = new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT).byteLength
          === Int32Array.BYTES_PER_ELEMENT
      }
      catch {
        // Some runtimes expose SharedArrayBuffer without allowing it in the current security context.
      }
    }
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
    const id = this.nextBodyId
    const positions = createPositions(descriptor.mesh.positions, this.sharedTransport)
    let sharedWithWorker = false
    if (!this.sharedTransport) {
      this.post({ descriptor, id, type: 'addBody' })
    }
    else {
      sharedWithWorker = this.postShared(
        { descriptor, id, sharedPositions: positions, type: 'addBody' },
        { descriptor, id, type: 'addBody' },
      )
    }
    this.bodies.set(id, { initial: positions.slice(), positions, sharedWithWorker })
    this.nextBodyId++
    return id
  }

  addCollider(descriptor: ColliderDescriptor): ColliderId {
    if (this.direct != null)
      return this.direct.addCollider(descriptor)
    this.assertActive()
    const id = this.nextColliderId
    this.post({ descriptor, id, type: 'addCollider' })
    this.nextColliderId++
    return id
  }

  addGrab(descriptor: GrabDescriptor): GrabId {
    if (this.direct != null)
      return this.direct.addGrab(descriptor)
    this.assertActive()
    this.requireBody(descriptor.body)
    const id = this.nextGrabId
    this.post({ descriptor, id, type: 'addGrab' })
    this.nextGrabId++
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
    this.post({ id, type: 'removeBody' })
    this.bodies.delete(id)
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
    this.post({ id, positions, type: 'resetBody' })
    // A shared body is updated by the Worker after all earlier commands. A
    // main-thread write here would race an in-flight solve and violate that
    // ordering. Transfer-mode mirrors are independent and still need updating.
    if (!body.sharedWithWorker || this.pending.size === 0)
      body.positions.set(source)
  }

  setMotionConstraintTargets(id: BodyId, positions: Float32Array): void {
    if (this.direct != null) {
      this.direct.setMotionConstraintTargets(id, positions)
      return
    }
    this.assertActive()
    const body = this.requireBody(id)
    if (positions.length !== body.positions.length)
      throw new RangeError('Motion targets must match the body particle count')
    this.post({ id, positions, type: 'setMotionConstraintTargets' })
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
      const signal = this.sharedTransport
        ? new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT))
        : undefined
      this.pending.set(requestId, { reject, resolve, signal })
      try {
        if (signal == null) {
          this.post({ delta, options, requestId, type: 'step' })
          return
        }
        if (this.postShared(
          { delta, options, requestId, signal, type: 'step' },
          { delta, options, requestId, type: 'step' },
        )) {
          void this.waitForSharedStep(requestId, signal)
        }
      }
      catch (error) {
        this.pending.delete(requestId)
        reject(error)
      }
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
    for (const pending of this.pending.values()) {
      pending.reject(error)
      if (pending.signal != null)
        Atomics.notify(pending.signal, 0)
    }
    this.pending.clear()
  }

  private handleMessage(message: WorkerResponse): void {
    // A WASM worker can finish its module handshake on an adapter that was
    // already attached to this backend during HMR. Control responses are not
    // step snapshots and must never fall through to the snapshot path.
    if (message.type === 'ready')
      return
    if (message.type !== 'step' && message.type !== 'error')
      return
    if (message.type === 'error') {
      const error = new Error(message.message)
      if (message.stack != null)
        error.stack = message.stack
      if (message.requestId == null) {
        this.fail(error)
        return
      }
      const pending = this.pending.get(message.requestId)
      pending?.reject(error)
      this.pending.delete(message.requestId)
      if (pending?.signal != null)
        Atomics.notify(pending.signal, 0)
      return
    }
    for (const [id, positions] of message.positions)
      this.bodies.get(id)?.positions.set(positions)
    const pending = this.pending.get(message.requestId)
    pending?.resolve()
    this.pending.delete(message.requestId)
    if (pending?.signal != null)
      Atomics.notify(pending.signal, 0)
  }

  private post(message: unknown): void {
    this.worker?.postMessage(message)
  }

  private postShared(message: unknown, fallback: unknown): boolean {
    try {
      this.post(message)
      return true
    }
    catch (error) {
      if (!isSharedTransportError(error))
        throw error
      this.sharedTransport = false
      this.post(fallback)
      return false
    }
  }

  private requireBody(id: BodyId): BodyMirror {
    const body = this.bodies.get(id)
    if (body == null)
      throw new RangeError(`Unknown cloth body ${id}`)
    return body
  }

  private async waitForSharedStep(requestId: number, signal: Int32Array): Promise<void> {
    while (this.pending.has(requestId)) {
      const completed = Atomics.load(signal, 0)
      if (completed !== 0) {
        this.pending.get(requestId)?.resolve()
        this.pending.delete(requestId)
        return
      }
      const wait = Atomics.waitAsync(signal, 0, 0)
      if (wait.async)
        await wait.value
    }
  }
}
