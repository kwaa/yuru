import type { BufferGeometry } from 'three'
import type {
  BodyId,
  ClothBodyDescriptor,
  ClothMeshData,
  ClothWorld,
  ClothWorldOptions,
  RuntimeDiagnostics,
} from 'yuru'

import { BufferAttribute, Matrix4, Mesh, SkinnedMesh, Vector3 } from 'three'
import { createClothWorld } from 'yuru'

import { YURU_VISUAL_VERTEX_MAP } from './skinned-cloth.js'

export { colliderFromThree, geometryToTriangleMeshShape } from './shapes.js'
export type { CapsuleLike, ThreeColliderShape, ThreeShapeOptions } from './shapes.js'

export type PinSelection = 'top' | false | readonly number[] | Uint32Array

export interface ThreeClothOptions extends Omit<ClothBodyDescriptor, 'mesh'> {
  inverseMasses?: Float32Array
  pin?: PinSelection
  pinTopRatio?: number
  simulationMesh?: Mesh
}

export interface ThreeKinematicClothColliderOptions extends Omit<ClothBodyDescriptor, 'mesh' | 'selfCollision'> {}

export interface ThreeMeshData {
  localRestPositions: Float32Array
  mesh: ClothMeshData
}

interface VisualBinding {
  indices: Uint32Array
  offsets: Float32Array
}

const geometryIndices = (geometry: BufferGeometry): Uint16Array | Uint32Array => {
  const position = geometry.getAttribute('position')
  const index = geometry.getIndex()
  if (index != null)
    return position.count > 0xFFFF ? Uint32Array.from(index.array) : Uint16Array.from(index.array)
  return position.count > 0xFFFF
    ? Uint32Array.from({ length: position.count }, (_, item) => item)
    : Uint16Array.from({ length: position.count }, (_, item) => item)
}

const connectedComponentRoots = (
  vertexCount: number,
  indices: Uint16Array | Uint32Array,
): Int32Array => {
  const parents = new Int32Array(vertexCount)
  for (let vertex = 0; vertex < vertexCount; vertex++)
    parents[vertex] = vertex
  const find = (vertex: number): number => {
    let root = vertex
    while (parents[root] !== root)
      root = parents[root]
    while (parents[vertex] !== vertex) {
      const parent = parents[vertex]
      parents[vertex] = root
      vertex = parent
    }
    return root
  }
  const join = (first: number, second: number): void => {
    const firstRoot = find(first)
    const secondRoot = find(second)
    if (firstRoot !== secondRoot)
      parents[secondRoot] = firstRoot
  }
  for (let offset = 0; offset < indices.length; offset += 3) {
    join(indices[offset], indices[offset + 1])
    join(indices[offset + 1], indices[offset + 2])
  }
  for (let vertex = 0; vertex < vertexCount; vertex++)
    parents[vertex] = find(vertex)
  return parents
}

const selectPins = (
  positions: Float32Array,
  indices: Uint16Array | Uint32Array,
  selection: PinSelection,
  topRatio: number,
): Uint32Array => {
  if (selection === false)
    return new Uint32Array()
  if (selection !== 'top')
    return Uint32Array.from(selection)
  const vertexCount = positions.length / 3
  const roots = connectedComponentRoots(vertexCount, indices)
  const minimums = new Float32Array(vertexCount).fill(Number.POSITIVE_INFINITY)
  const maximums = new Float32Array(vertexCount).fill(Number.NEGATIVE_INFINITY)
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const root = roots[vertex]
    const y = positions[vertex * 3 + 1]
    minimums[root] = Math.min(minimums[root], y)
    maximums[root] = Math.max(maximums[root], y)
  }
  const result: number[] = []
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const root = roots[vertex]
    const threshold = maximums[root] - (maximums[root] - minimums[root]) * topRatio
    if (positions[vertex * 3 + 1] >= threshold)
      result.push(vertex)
  }
  return Uint32Array.from(result)
}

const pinnedFromMasses = (inverseMasses: Float32Array): Uint32Array => {
  const result: number[] = []
  for (let index = 0; index < inverseMasses.length; index++) {
    if (inverseMasses[index] === 0)
      result.push(index)
  }
  return Uint32Array.from(result)
}

const readWorldPositions = (mesh: Mesh, target?: Float32Array): Float32Array => {
  mesh.updateWorldMatrix(true, false)
  const position = mesh.geometry.getAttribute('position')
  if (position == null)
    throw new Error(`${mesh.name || 'Mesh'} requires a position attribute`)
  const result = target ?? new Float32Array(position.count * 3)
  if (result.length !== position.count * 3)
    throw new RangeError('World-position target does not match the mesh vertex count')
  const point = new Vector3()
  for (let index = 0; index < position.count; index++) {
    mesh.getVertexPosition(index, point)
    mesh.localToWorld(point)
    point.toArray(result, index * 3)
  }
  return result
}

const arraysEqual = (first: Float32Array, second: Float32Array): boolean => {
  if (first.length !== second.length)
    return false
  for (let index = 0; index < first.length; index++) {
    if (first[index] !== second[index])
      return false
  }
  return true
}

const mappedVisualBinding = (
  visualPositions: Float32Array,
  simulationPositions: Float32Array,
  directIndices: Uint32Array,
): VisualBinding => {
  const visualCount = visualPositions.length / 3
  const simulationCount = simulationPositions.length / 3
  if (directIndices.length !== visualCount)
    throw new RangeError('Visual vertex map does not match the display mesh vertex count')
  const indices = new Uint32Array(visualCount)
  const offsets = new Float32Array(visualPositions.length)
  for (let index = 0; index < visualCount; index++) {
    const simulationIndex = directIndices[index]
    if (simulationIndex >= simulationCount)
      throw new RangeError('Visual vertex map references a missing simulation particle')
    indices[index] = simulationIndex
    const visualOffset = index * 3
    const simulationOffset = simulationIndex * 3
    offsets[visualOffset] = visualPositions[visualOffset] - simulationPositions[simulationOffset]
    offsets[visualOffset + 1] = visualPositions[visualOffset + 1] - simulationPositions[simulationOffset + 1]
    offsets[visualOffset + 2] = visualPositions[visualOffset + 2] - simulationPositions[simulationOffset + 2]
  }
  return { indices, offsets }
}

const nearestVisualBinding = (
  visualPositions: Float32Array,
  simulationPositions: Float32Array,
): VisualBinding => {
  const visualCount = visualPositions.length / 3
  const simulationCount = simulationPositions.length / 3
  const indices = new Uint32Array(visualCount)
  const offsets = new Float32Array(visualPositions.length)
  if (arraysEqual(visualPositions, simulationPositions)) {
    for (let index = 0; index < visualCount; index++)
      indices[index] = index
    return { indices, offsets }
  }
  for (let visualIndex = 0; visualIndex < visualCount; visualIndex++) {
    const visualOffset = visualIndex * 3
    const x = visualPositions[visualOffset]
    const y = visualPositions[visualOffset + 1]
    const z = visualPositions[visualOffset + 2]
    let closest = 0
    let closestDistance = Number.POSITIVE_INFINITY
    for (let simulationIndex = 0; simulationIndex < simulationCount; simulationIndex++) {
      const simulationOffset = simulationIndex * 3
      const dx = x - simulationPositions[simulationOffset]
      const dy = y - simulationPositions[simulationOffset + 1]
      const dz = z - simulationPositions[simulationOffset + 2]
      const distance = dx * dx + dy * dy + dz * dz
      if (distance < closestDistance) {
        closest = simulationIndex
        closestDistance = distance
      }
    }
    indices[visualIndex] = closest
    const simulationOffset = closest * 3
    offsets[visualOffset] = x - simulationPositions[simulationOffset]
    offsets[visualOffset + 1] = y - simulationPositions[simulationOffset + 1]
    offsets[visualOffset + 2] = z - simulationPositions[simulationOffset + 2]
  }
  return { indices, offsets }
}

const createVisualBinding = (
  visualPositions: Float32Array,
  simulationPositions: Float32Array,
  directIndices?: Uint32Array,
): VisualBinding => {
  if (simulationPositions.length === 0)
    throw new Error('A simulation mesh must contain at least one vertex')
  return directIndices == null
    ? nearestVisualBinding(visualPositions, simulationPositions)
    : mappedVisualBinding(visualPositions, simulationPositions, directIndices)
}

export const meshToClothData = (
  mesh: Mesh,
  options: Pick<ThreeClothOptions, 'inverseMasses' | 'pin' | 'pinTopRatio'> = {},
): ThreeMeshData => {
  mesh.updateWorldMatrix(true, false)
  const position = mesh.geometry.getAttribute('position')
  if (position == null)
    throw new Error(`${mesh.name || 'Mesh'} requires a position attribute`)
  const worldPositions = new Float32Array(position.count * 3)
  const localRestPositions = new Float32Array(position.count * 3)
  const indices = geometryIndices(mesh.geometry)
  const point = new Vector3()
  for (let index = 0; index < position.count; index++) {
    mesh.getVertexPosition(index, point)
    point.toArray(localRestPositions, index * 3)
    mesh.localToWorld(point)
    point.toArray(worldPositions, index * 3)
  }
  const inverseMasses = options.inverseMasses?.slice() ?? new Float32Array(position.count).fill(1)
  const pinned = selectPins(worldPositions, indices, options.pin ?? 'top', options.pinTopRatio ?? 0.08)
  for (const index of pinned)
    inverseMasses[index] = 0
  return {
    localRestPositions,
    mesh: {
      indices,
      inverseMasses,
      positions: worldPositions,
    },
  }
}

const bakeSkinnedDisplay = (source: SkinnedMesh, localRestPositions: Float32Array): Mesh => {
  const geometry = source.geometry.clone()
  geometry.setAttribute('position', new BufferAttribute(localRestPositions.slice(), 3))
  geometry.computeVertexNormals()
  const display = new Mesh(geometry, source.material)
  display.name = `${source.name || 'SkinnedMesh'}_YuruDisplay`
  display.position.copy(source.position)
  display.quaternion.copy(source.quaternion)
  display.scale.copy(source.scale)
  display.renderOrder = source.renderOrder
  display.castShadow = source.castShadow
  display.receiveShadow = source.receiveShadow
  display.frustumCulled = false
  source.parent?.add(display)
  return display
}

const isSkinnedMesh = (mesh: Mesh): mesh is SkinnedMesh => mesh instanceof SkinnedMesh

export class ThreeClothController {
  readonly body: BodyId
  readonly mesh: Mesh
  readonly pinnedIndices: Uint32Array
  readonly sourceMesh: Mesh
  readonly status = 'ready' as const

  private disposed = false
  private readonly localRestPositions: Float32Array
  private readonly motionTargetPositions?: Float32Array
  private readonly owner: ThreeYuruWorld
  private readonly ownsDisplay: boolean
  private readonly simulationSource: Mesh
  private readonly sourceWasVisible: boolean
  private readonly syncPoint = new Vector3()
  private readonly targetPoint = new Vector3()
  private readonly targetPositions: Float32Array
  private readonly visualBinding: VisualBinding
  private readonly worldToLocal = new Matrix4()

  constructor(
    owner: ThreeYuruWorld,
    sourceMesh: Mesh,
    options: ThreeClothOptions = {},
  ) {
    this.owner = owner
    this.sourceMesh = sourceMesh
    const simulationSource = options.simulationMesh ?? sourceMesh
    this.simulationSource = simulationSource
    const data = meshToClothData(simulationSource, options)
    this.localRestPositions = data.localRestPositions
    this.pinnedIndices = pinnedFromMasses(data.mesh.inverseMasses!)
    this.targetPositions = new Float32Array(this.pinnedIndices.length * 3)
    this.sourceWasVisible = sourceMesh.visible
    this.ownsDisplay = isSkinnedMesh(sourceMesh)
    this.mesh = isSkinnedMesh(sourceMesh)
      ? bakeSkinnedDisplay(sourceMesh, meshToClothData(sourceMesh, { pin: false }).localRestPositions)
      : sourceMesh
    const proxyData = simulationSource.geometry.userData as Record<string, unknown>
    const directIndices = proxyData[YURU_VISUAL_VERTEX_MAP]
    this.visualBinding = createVisualBinding(
      readWorldPositions(this.mesh),
      data.mesh.positions,
      directIndices instanceof Uint32Array ? directIndices : undefined,
    )
    if (this.ownsDisplay)
      sourceMesh.visible = false
    const bodyOptions = { ...options }
    delete bodyOptions.inverseMasses
    delete bodyOptions.pin
    delete bodyOptions.pinTopRatio
    delete bodyOptions.simulationMesh
    if (bodyOptions.motionConstraints !== false && bodyOptions.motionConstraints != null) {
      this.motionTargetPositions = data.mesh.positions.slice()
      bodyOptions.motionConstraints = {
        ...bodyOptions.motionConstraints,
        targets: this.motionTargetPositions,
      }
    }
    this.body = owner.core.addBody({
      ...bodyOptions,
      mesh: data.mesh,
    })
  }

  dispose(): void {
    if (this.disposed)
      return
    this.disposed = true
    this.owner.detach(this)
    this.owner.core.removeBody(this.body)
    if (this.ownsDisplay) {
      this.mesh.removeFromParent()
      this.mesh.geometry.dispose()
      this.sourceMesh.visible = this.sourceWasVisible
    }
  }

  reset(): void {
    this.owner.core.resetBody(this.body)
    this.syncVisual()
  }

  syncVisual(): void {
    if (this.disposed)
      return
    this.mesh.updateWorldMatrix(true, false)
    this.worldToLocal.copy(this.mesh.matrixWorld).invert()
    const positions = this.owner.core.getPositions(this.body)
    const attribute = this.mesh.geometry.getAttribute('position')
    const point = this.syncPoint
    for (let index = 0; index < attribute.count; index++) {
      const bodyOffset = this.visualBinding.indices[index] * 3
      const visualOffset = index * 3
      point.set(
        positions[bodyOffset] + this.visualBinding.offsets[visualOffset],
        positions[bodyOffset + 1] + this.visualBinding.offsets[visualOffset + 1],
        positions[bodyOffset + 2] + this.visualBinding.offsets[visualOffset + 2],
      )
      point.applyMatrix4(this.worldToLocal)
      attribute.setXYZ(index, point.x, point.y, point.z)
    }
    attribute.needsUpdate = true
    this.mesh.geometry.computeVertexNormals()
  }

  async update(delta: number): Promise<void> {
    return this.owner.update(delta)
  }

  updateKinematicTargets(): void {
    if (this.disposed)
      return
    if (this.motionTargetPositions != null) {
      readWorldPositions(this.simulationSource, this.motionTargetPositions)
      this.owner.core.setMotionConstraintTargets(this.body, this.motionTargetPositions)
    }
    if (this.pinnedIndices.length === 0)
      return
    this.simulationSource.updateWorldMatrix(true, false)
    const targets = this.targetPositions
    const point = this.targetPoint
    for (let item = 0; item < this.pinnedIndices.length; item++) {
      const index = this.pinnedIndices[item]
      if (isSkinnedMesh(this.simulationSource))
        this.simulationSource.getVertexPosition(index, point)
      else
        point.fromArray(this.localRestPositions, index * 3)
      this.simulationSource.localToWorld(point)
      point.toArray(targets, item * 3)
    }
    this.owner.core.setParticleTargets(this.body, this.pinnedIndices, targets)
  }
}

/** A skinned or transformed mesh that participates in cloth inter-collision without being simulated or rendered by Yuru. */
export class ThreeKinematicClothCollider {
  readonly body: BodyId
  readonly mesh: Mesh

  private disposed = false
  private readonly indices: Uint32Array
  private readonly owner: ThreeYuruWorld
  private readonly targets: Float32Array

  constructor(owner: ThreeYuruWorld, mesh: Mesh, options: ThreeKinematicClothColliderOptions = {}) {
    this.owner = owner
    this.mesh = mesh
    const data = meshToClothData(mesh, { pin: false })
    const particleCount = data.mesh.positions.length / 3
    data.mesh.inverseMasses = new Float32Array(particleCount)
    this.indices = Uint32Array.from({ length: particleCount }, (_, item) => item)
    this.targets = data.mesh.positions.slice()
    this.body = owner.core.addBody({
      ...options,
      mesh: data.mesh,
      selfCollision: false,
    })
  }

  dispose(): void {
    if (this.disposed)
      return
    this.disposed = true
    this.owner.detachKinematicCollider(this)
    this.owner.core.removeBody(this.body)
  }

  updateKinematicTargets(): void {
    if (this.disposed)
      return
    readWorldPositions(this.mesh, this.targets)
    this.owner.core.setParticleTargets(this.body, this.indices, this.targets)
  }
}

export class ThreeYuruWorld {
  readonly core: ClothWorld
  get diagnostics(): Readonly<RuntimeDiagnostics> {
    return this.core.diagnostics
  }

  private readonly beforeStep = new Set<() => void>()
  private readonly controllers = new Set<ThreeClothController>()
  private readonly kinematicColliders = new Set<ThreeKinematicClothCollider>()

  private pending = Promise.resolve()
  private queuedDelta = 0
  private updating = false

  constructor(options: ClothWorldOptions = {}) {
    this.core = createClothWorld(options)
  }

  addBeforeStep(callback: () => void): () => void {
    this.beforeStep.add(callback)
    return () => this.beforeStep.delete(callback)
  }

  attachCloth(mesh: Mesh, options: ThreeClothOptions = {}): ThreeClothController {
    const controller = new ThreeClothController(this, mesh, options)
    this.controllers.add(controller)
    return controller
  }

  attachKinematicClothCollider(
    mesh: Mesh,
    options: ThreeKinematicClothColliderOptions = {},
  ): ThreeKinematicClothCollider {
    const controller = new ThreeKinematicClothCollider(this, mesh, options)
    this.kinematicColliders.add(controller)
    return controller
  }

  detach(controller: ThreeClothController): void {
    this.controllers.delete(controller)
  }

  detachKinematicCollider(controller: ThreeKinematicClothCollider): void {
    this.kinematicColliders.delete(controller)
  }

  dispose(): void {
    for (const controller of [...this.controllers])
      controller.dispose()
    for (const collider of [...this.kinematicColliders])
      collider.dispose()
    this.beforeStep.clear()
    this.core.dispose()
  }

  async update(delta: number): Promise<void> {
    if (delta < 0 || !Number.isFinite(delta))
      throw new RangeError('Frame delta must be a non-negative finite number')
    // Rendering should never enter a catch-up spiral when simulation is slower
    // than a frame. ClothWorld still supports catch-up for explicit core users.
    this.queuedDelta = Math.min(this.queuedDelta + delta, this.core.fixedDelta)
    if (!this.updating) {
      this.updating = true
      this.pending = this.drainUpdates().finally(() => {
        this.updating = false
      })
    }
    return this.pending
  }

  private async drainUpdates(): Promise<void> {
    while (this.queuedDelta > 0) {
      const delta = this.queuedDelta
      this.queuedDelta = 0
      for (const callback of this.beforeStep)
        callback()
      for (const collider of this.kinematicColliders)
        collider.updateKinematicTargets()
      for (const controller of this.controllers)
        controller.updateKinematicTargets()
      await this.core.step(delta)
      for (const controller of this.controllers)
        controller.syncVisual()
    }
  }
}

export const createThreeYuruWorld = (options: ClothWorldOptions = {}): ThreeYuruWorld => new ThreeYuruWorld(options)

export interface TSLProbeResult {
  reason?: string
  supported: boolean
}

/** Capability probe kept separate from backend creation: no hidden fallback. */
export const probeTSLBackend = (renderer: { backend?: { isWebGPUBackend?: boolean }, isWebGPURenderer?: boolean }): TSLProbeResult => {
  if (!renderer.isWebGPURenderer)
    return { reason: 'A Three WebGPURenderer is required.', supported: false }
  if (renderer.backend?.isWebGPUBackend === false)
    return { reason: 'WebGPURenderer is using its WebGL backend; TSL compute requires WebGPU.', supported: false }
  return { supported: true }
}
