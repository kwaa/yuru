import type {
  BodyId,
  ClothBackend,
  ClothBodyDescriptor,
  ClothSpeedLimit,
  ClothWorldOptions,
  ColliderDescriptor,
  ColliderId,
  ForceField,
  GrabDescriptor,
  GrabId,
  QualityPreset,
  QualityProfile,
  RuntimeDiagnostics,
  Vec3,
  Vec3Like,
} from './types.js'

import { CPUBackend } from './cpu-backend.js'
import { readVec3 } from './math.js'
import { QUALITY_PRESETS } from './types.js'

const now = (): number => globalThis.performance?.now() ?? Date.now()

const resolveQuality = (quality?: Partial<QualityProfile> | QualityPreset): QualityProfile => {
  if (typeof quality === 'string')
    return { ...QUALITY_PRESETS[quality] }
  return { ...QUALITY_PRESETS.medium, ...quality }
}

const resolveSpeedLimit = (speedLimit: ClothSpeedLimit = 'automatic'): ClothSpeedLimit => {
  if (typeof speedLimit === 'number' && (speedLimit <= 0 || !Number.isFinite(speedLimit)))
    throw new RangeError('A numeric cloth speed limit must be a positive finite value in meters per second')
  return speedLimit
}

export class ClothWorld {
  readonly backend: ClothBackend
  readonly fixedDelta: number

  get diagnostics(): Readonly<RuntimeDiagnostics> {
    return {
      ...this.diagnosticsState,
      backend: { ...this.diagnosticsState.backend },
      fallbackReasons: [...this.diagnosticsState.fallbackReasons],
      quality: { ...this.diagnosticsState.quality },
    }
  }

  private accumulator = 0
  private readonly diagnosticsState: RuntimeDiagnostics
  private disposed = false
  private forceFields: readonly ForceField[]
  private gravity: Vec3
  private pending = Promise.resolve()

  private quality: QualityProfile
  private speedLimit: ClothSpeedLimit

  constructor(
    backend: ClothBackend,
    options: ClothWorldOptions,
  ) {
    this.backend = backend
    this.fixedDelta = options.fixedDelta ?? 1 / 60
    if (this.fixedDelta <= 0 || !Number.isFinite(this.fixedDelta))
      throw new RangeError('fixedDelta must be a positive finite number')
    this.gravity = readVec3(options.gravity ?? [0, -9.81, 0])
    this.forceFields = options.forceFields ?? []
    this.quality = resolveQuality(options.quality)
    this.speedLimit = resolveSpeedLimit(options.speedLimit)
    this.diagnosticsState = {
      averageStepMs: 0,
      backend: backend.capabilities,
      droppedTime: 0,
      fallbackReasons: [],
      lastStepMs: 0,
      quality: { ...this.quality },
      simulatedSteps: 0,
      speedLimit: this.speedLimit,
    }
  }

  addBody(descriptor: ClothBodyDescriptor): BodyId {
    this.assertActive()
    return this.backend.addBody(descriptor)
  }

  addCollider(descriptor: ColliderDescriptor): ColliderId {
    this.assertActive()
    return this.backend.addCollider(descriptor)
  }

  addGrab(descriptor: GrabDescriptor): GrabId {
    this.assertActive()
    return this.backend.addGrab(descriptor)
  }

  dispose(): void {
    if (this.disposed)
      return
    this.disposed = true
    this.backend.dispose()
  }

  getPositions(id: BodyId): Float32Array {
    this.assertActive()
    return this.backend.getPositions(id)
  }

  async idle(): Promise<void> {
    await this.pending
  }

  removeBody(id: BodyId): void {
    this.assertActive()
    this.backend.removeBody(id)
  }

  removeCollider(id: ColliderId): void {
    this.assertActive()
    this.backend.removeCollider(id)
  }

  removeGrab(id: GrabId): void {
    this.assertActive()
    this.backend.removeGrab(id)
  }

  resetBody(id: BodyId, positions?: Float32Array): void {
    this.assertActive()
    this.backend.resetBody(id, positions)
  }

  setForceFields(forceFields: readonly ForceField[]): void {
    this.forceFields = forceFields
  }

  setGravity(gravity: Vec3Like): void {
    this.gravity = readVec3(gravity)
  }

  setParticleTargets(id: BodyId, indices: Uint32Array, positions: Float32Array): void {
    this.assertActive()
    this.backend.setParticleTargets(id, indices, positions)
  }

  setQuality(quality: Partial<QualityProfile> | QualityPreset): void {
    this.quality = resolveQuality(quality)
    this.diagnosticsState.quality = { ...this.quality }
  }

  setSpeedLimit(speedLimit: ClothSpeedLimit): void {
    this.speedLimit = resolveSpeedLimit(speedLimit)
    this.diagnosticsState.speedLimit = this.speedLimit
  }

  /**
   * Adds render time to the fixed-step accumulator. Calls are serialized so a
   * Worker-backed backend can lag by a frame without corrupting state.
   */
  async step(delta: number): Promise<number> {
    this.assertActive()
    if (delta < 0 || !Number.isFinite(delta))
      throw new RangeError('Frame delta must be a non-negative finite number')
    this.pending = this.pending.then(async () => {
      this.accumulator += delta
      let steps = 0
      const started = now()
      while (this.accumulator >= this.fixedDelta && steps < this.quality.maxCatchUpSteps) {
        await this.backend.step(this.fixedDelta, {
          forceFields: this.forceFields,
          gravity: this.gravity,
          quality: this.quality,
          speedLimit: this.speedLimit,
        })
        this.accumulator -= this.fixedDelta
        steps++
      }
      if (this.accumulator >= this.fixedDelta) {
        const retained = this.accumulator % this.fixedDelta
        this.diagnosticsState.droppedTime += this.accumulator - retained
        this.accumulator = retained
      }
      const elapsed = now() - started
      this.diagnosticsState.simulatedSteps += steps
      this.diagnosticsState.lastStepMs = elapsed
      const samples = this.diagnosticsState.simulatedSteps
      if (steps > 0)
        this.diagnosticsState.averageStepMs += (elapsed / steps - this.diagnosticsState.averageStepMs) / Math.max(1, samples)
    })
    return this.pending.then(() => this.accumulator / this.fixedDelta)
  }

  updateCollider(id: ColliderId, descriptor: ColliderDescriptor): void {
    this.assertActive()
    this.backend.updateCollider(id, descriptor)
  }

  updateGrab(id: GrabId, position: Vec3Like): void {
    this.assertActive()
    this.backend.updateGrab(id, position)
  }

  private assertActive(): void {
    if (this.disposed)
      throw new Error('ClothWorld has been disposed')
  }
}

export const createClothWorld = (options: ClothWorldOptions = {}): ClothWorld =>
  new ClothWorld(options.backend ?? new CPUBackend(), options)
