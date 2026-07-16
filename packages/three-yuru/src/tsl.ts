import type { ComputeNode, UniformNode, WebGPURenderer } from 'three/webgpu'
import type { CpuIntegrationKernel } from 'yuru'

import { float, Fn, If, instanceIndex, storage, uniform } from 'three/tsl'
import { StorageBufferAttribute } from 'three/webgpu'
import { CPUBackend } from 'yuru'

interface PackedBuffers {
  accelerations: Float32Array
  inverseMasses: Float32Array
  positions: Float32Array
  previous: Float32Array
}

const packVec3 = (source: Float32Array, target: Float32Array): void => {
  for (let index = 0; index < source.length / 3; index++) {
    const sourceOffset = index * 3
    const targetOffset = index * 4
    target[targetOffset] = source[sourceOffset]
    target[targetOffset + 1] = source[sourceOffset + 1]
    target[targetOffset + 2] = source[sourceOffset + 2]
    target[targetOffset + 3] = 0
  }
}

const unpackVec3 = (source: Float32Array, target: Float32Array): void => {
  for (let index = 0; index < target.length / 3; index++) {
    const sourceOffset = index * 4
    const targetOffset = index * 3
    target[targetOffset] = source[sourceOffset]
    target[targetOffset + 1] = source[sourceOffset + 1]
    target[targetOffset + 2] = source[sourceOffset + 2]
  }
}

class TSLIntegrationSession {
  readonly accelerationAttribute: StorageBufferAttribute
  readonly buffers: PackedBuffers
  readonly compute: ComputeNode
  readonly damping: UniformNode<'float', number>
  readonly delta: UniformNode<'float', number>
  readonly inverseMassAttribute: StorageBufferAttribute
  readonly positionAttribute: StorageBufferAttribute
  readonly previousAttribute: StorageBufferAttribute

  constructor(count: number) {
    this.buffers = {
      accelerations: new Float32Array(count * 4),
      inverseMasses: new Float32Array(count),
      positions: new Float32Array(count * 4),
      previous: new Float32Array(count * 4),
    }
    this.positionAttribute = new StorageBufferAttribute(this.buffers.positions, 4)
    this.previousAttribute = new StorageBufferAttribute(this.buffers.previous, 4)
    this.inverseMassAttribute = new StorageBufferAttribute(this.buffers.inverseMasses, 1)
    this.accelerationAttribute = new StorageBufferAttribute(this.buffers.accelerations, 4)
    this.delta = uniform(0)
    this.damping = uniform(0)

    const positions = storage(this.positionAttribute, 'vec4', count)
    const previous = storage(this.previousAttribute, 'vec4', count)
    const inverseMasses = storage(this.inverseMassAttribute, 'float', count)
    const accelerations = storage(this.accelerationAttribute, 'vec4', count)
    this.compute = Fn(() => {
      const position = positions.element(instanceIndex)
      const priorPosition = previous.element(instanceIndex)
      const current = position.toVar()
      const prior = priorPosition.toVar()
      If(inverseMasses.element(instanceIndex).greaterThan(0), () => {
        const velocity = current.sub(prior).mul(float(1).sub(this.damping))
        priorPosition.assign(current)
        position.assign(current.add(velocity).add(accelerations.element(instanceIndex).mul(this.delta.mul(this.delta))))
      })
    })().compute(count)
  }

  dispose(): void {
    this.compute.dispose()
  }
}

/**
 * WebGPU/TSL prediction kernel. XPBD constraints and collision handling remain
 * in the typed-array CPU solver, so this is useful for validating TSL execution but
 * avoids pretending that CPU/GPU readback is the final high-performance path.
 */
export class TSLIntegrationKernel implements CpuIntegrationKernel {
  readonly capabilities = {
    continuousCollision: true,
    kind: 'tsl' as const,
    simd: false,
    threads: false,
    worker: false,
  }

  private readonly renderer: WebGPURenderer
  private readonly sessions = new Set<TSLIntegrationSession>()
  private readonly sessionsByPositions = new WeakMap<Float32Array, TSLIntegrationSession>()

  constructor(renderer: WebGPURenderer) {
    this.renderer = renderer
  }

  dispose(): void {
    for (const session of this.sessions)
      session.dispose()
    this.sessions.clear()
  }

  async integrate(
    positions: Float32Array,
    previous: Float32Array,
    inverseMasses: Float32Array,
    accelerations: Float32Array,
    delta: number,
    damping: number,
  ): Promise<void> {
    let session = this.sessionsByPositions.get(positions)
    if (session == null) {
      session = new TSLIntegrationSession(inverseMasses.length)
      this.sessionsByPositions.set(positions, session)
      this.sessions.add(session)
    }
    packVec3(positions, session.buffers.positions)
    packVec3(previous, session.buffers.previous)
    packVec3(accelerations, session.buffers.accelerations)
    session.buffers.inverseMasses.set(inverseMasses)
    session.positionAttribute.needsUpdate = true
    session.previousAttribute.needsUpdate = true
    session.accelerationAttribute.needsUpdate = true
    session.inverseMassAttribute.needsUpdate = true
    session.delta.value = delta
    session.damping.value = damping

    await this.renderer.computeAsync(session.compute)
    const [positionBuffer, previousBuffer] = await Promise.all([
      this.renderer.getArrayBufferAsync(session.positionAttribute),
      this.renderer.getArrayBufferAsync(session.previousAttribute),
    ])
    unpackVec3(new Float32Array(positionBuffer), positions)
    unpackVec3(new Float32Array(previousBuffer), previous)
  }
}

export const createTSLBackend = (renderer: WebGPURenderer): CPUBackend => {
  const backend = renderer.backend as typeof renderer.backend & { isWebGPUBackend?: boolean }
  if (!renderer.isWebGPURenderer || backend.isWebGPUBackend === false)
    throw new Error('createTSLBackend requires a Three WebGPURenderer using its WebGPU backend')
  return new CPUBackend({ integrationKernel: new TSLIntegrationKernel(renderer), worker: false })
}
