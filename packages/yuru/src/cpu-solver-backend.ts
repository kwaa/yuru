import type {
  BackendCapabilities,
  BackendStepOptions,
  BodyId,
  ClothBackend,
  ClothBodyDescriptor,
  ClothMaterial,
  ClothSpeedLimit,
  ColliderDescriptor,
  ColliderId,
  CollisionFilter,
  CpuIntegrationKernel,
  ForceField,
  GrabDescriptor,
  GrabId,
  Vec3,
  Vec3Like,
} from './types.js'

import { closestPointOnSegment, closestPointOnTriangle, EPSILON, inverseRotate, normalize, readQuat, readVec3, rotate } from './math.js'
import { buildTopology, isWithinTwoRings, STRETCH_EDGE } from './topology.js'
import { DEFAULT_CLOTH_MATERIAL, DEFAULT_COLLISION_FILTER } from './types.js'

interface BodyState {
  accelerations: Float32Array
  collisionCellSize: number
  collisionThickness: number
  filter: CollisionFilter
  id: BodyId
  indices: Uint16Array | Uint32Array
  initial: Float32Array
  inverseMasses: Float32Array
  label?: string
  materials: ClothMaterial[]
  maximumSelfCollisionDepenetration: number
  maximumSelfCollisionDisplacement: number
  particleSpacing: number
  positions: Float32Array
  previous: Float32Array
  selfCollision: boolean
  targets?: ParticleTargets
  topology: ReturnType<typeof buildTopology>
  triangleMaterialIndices?: Uint16Array
  velocityScratch: Float32Array
}

interface ColliderState {
  descriptor: ColliderDescriptor
  filter: CollisionFilter
  id: ColliderId
}

interface GrabState {
  body: BodyId
  compliance: number
  id: GrabId
  offsets: Float32Array
  particles: Uint32Array
  position: Vec3
}

interface ParticleTargets {
  indices: Uint32Array
  positions: Float32Array
}

const mergeFilter = (filter?: Partial<CollisionFilter>): CollisionFilter => ({
  group: filter?.group ?? DEFAULT_COLLISION_FILTER.group,
  mask: filter?.mask ?? DEFAULT_COLLISION_FILTER.mask,
})

const filtersCollide = (a: CollisionFilter, b: CollisionFilter): boolean =>
  ((a.mask & b.group) >>> 0) !== 0 && ((b.mask & a.group) >>> 0) !== 0

const mergeMaterial = (material?: Partial<ClothMaterial>): ClothMaterial => ({
  ...DEFAULT_CLOTH_MATERIAL,
  ...material,
})

const materialForTriangle = (body: BodyState, triangle: number): ClothMaterial =>
  body.materials[body.triangleMaterialIndices?.[triangle] ?? 0] ?? body.materials[0]

const cellKey = (x: number, y: number, z: number, inverseCellSize: number): number => {
  const cellX = Math.floor(x * inverseCellSize)
  const cellY = Math.floor(y * inverseCellSize)
  const cellZ = Math.floor(z * inverseCellSize)
  return ((cellX * 73_856_093) ^ (cellY * 19_349_663) ^ (cellZ * 83_492_791)) >>> 0
}

const particleSpacing = (topology: ReturnType<typeof buildTopology>): number => {
  let edgeLength = 0
  let edgeCount = 0
  for (let edge = 0; edge < topology.edgeKinds.length; edge++) {
    if (topology.edgeKinds[edge] !== STRETCH_EDGE)
      continue
    edgeLength += topology.edgeRestLengths[edge]
    edgeCount++
  }
  return edgeCount === 0 ? 0 : edgeLength / edgeCount
}

const collisionCellSize = (averageEdgeLength: number, materials: readonly ClothMaterial[]): number => {
  const thickness = Math.max(...materials.map(material => material.thickness))
  return Math.max(0.01, thickness * 3, averageEdgeLength * 2)
}

const limitedDepenetration = (penetration: number, initialPenetration: number, maximumCorrection: number): number => {
  if (!Number.isFinite(maximumCorrection) || initialPenetration <= 0)
    return penetration
  const predictedPenetration = Math.max(0, penetration - initialPenetration)
  return Math.min(penetration, predictedPenetration + Math.min(initialPenetration, maximumCorrection))
}

const hashNoise = (x: number, y: number, z: number, seed: number): number => {
  const value = Math.sin(x * 12.9898 + y * 78.233 + z * 37.719 + seed * 19.19) * 43758.5453
  return (value - Math.floor(value)) * 2 - 1
}

const closestSegmentSegment = (
  p1: Vec3,
  q1: Vec3,
  p2: Vec3,
  q2: Vec3,
): [number, number, number, number, number, number, number, number] => {
  const d1x = q1[0] - p1[0]
  const d1y = q1[1] - p1[1]
  const d1z = q1[2] - p1[2]
  const d2x = q2[0] - p2[0]
  const d2y = q2[1] - p2[1]
  const d2z = q2[2] - p2[2]
  const rx = p1[0] - p2[0]
  const ry = p1[1] - p2[1]
  const rz = p1[2] - p2[2]
  const a = d1x * d1x + d1y * d1y + d1z * d1z
  const e = d2x * d2x + d2y * d2y + d2z * d2z
  const f = d2x * rx + d2y * ry + d2z * rz
  let s = 0
  let t = 0

  if (a <= EPSILON && e <= EPSILON) {
    // Both segments degenerate to points.
  }
  else if (a <= EPSILON) {
    t = Math.max(0, Math.min(1, f / e))
  }
  else {
    const c = d1x * rx + d1y * ry + d1z * rz
    if (e <= EPSILON) {
      s = Math.max(0, Math.min(1, -c / a))
    }
    else {
      const b = d1x * d2x + d1y * d2y + d1z * d2z
      const denominator = a * e - b * b
      if (denominator !== 0)
        s = Math.max(0, Math.min(1, (b * f - c * e) / denominator))
      t = (b * s + f) / e
      if (t < 0) {
        t = 0
        s = Math.max(0, Math.min(1, -c / a))
      }
      else if (t > 1) {
        t = 1
        s = Math.max(0, Math.min(1, (b - c) / a))
      }
    }
  }

  return [
    p1[0] + d1x * s,
    p1[1] + d1y * s,
    p1[2] + d1z * s,
    p2[0] + d2x * t,
    p2[1] + d2y * t,
    p2[2] + d2z * t,
    s,
    t,
  ]
}

/** Typed-array Small Steps XPBD solver used by CPU and numeric-kernel backends. */
export class CPUSolverBackend implements ClothBackend {
  readonly capabilities: BackendCapabilities

  private readonly bodies = new Map<BodyId, BodyState>()
  private readonly colliders = new Map<ColliderId, ColliderState>()
  private elapsed = 0
  private readonly grabs = new Map<GrabId, GrabState>()
  private readonly integrationKernel?: CpuIntegrationKernel
  private nextBodyId = 1
  private nextColliderId = 1
  private nextGrabId = 1

  constructor(integrationKernel?: CpuIntegrationKernel) {
    this.integrationKernel = integrationKernel
    this.capabilities = integrationKernel?.capabilities ?? {
      continuousCollision: true,
      kind: 'cpu',
      simd: false,
      threads: false,
      worker: false,
    }
  }

  addBody(descriptor: ClothBodyDescriptor): BodyId {
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

    const positions = mesh.positions.slice()
    const materials = descriptor.materials != null && descriptor.materials.length > 0
      ? descriptor.materials.map(mergeMaterial)
      : [mergeMaterial()]
    const topology = buildTopology(positions, mesh.indices)
    const spacing = particleSpacing(topology)
    const thickness = Math.max(...materials.map(material => material.thickness))
    const id = this.nextBodyId++
    this.bodies.set(id, {
      accelerations: new Float32Array(positions.length),
      collisionCellSize: collisionCellSize(spacing, materials),
      collisionThickness: thickness,
      filter: mergeFilter(descriptor.collisionFilter),
      id,
      indices: mesh.indices.slice(),
      initial: positions.slice(),
      inverseMasses: mesh.inverseMasses?.slice() ?? new Float32Array(count).fill(1),
      label: descriptor.id,
      materials,
      maximumSelfCollisionDepenetration: Number.POSITIVE_INFINITY,
      maximumSelfCollisionDisplacement: Number.POSITIVE_INFINITY,
      particleSpacing: spacing,
      positions,
      previous: positions.slice(),
      selfCollision: descriptor.selfCollision ?? true,
      topology,
      triangleMaterialIndices: mesh.triangleMaterialIndices?.slice(),
      velocityScratch: new Float32Array(positions.length),
    })
    return id
  }

  addCollider(descriptor: ColliderDescriptor): ColliderId {
    const id = this.nextColliderId++
    this.colliders.set(id, { descriptor, filter: mergeFilter(descriptor.collisionFilter), id })
    return id
  }

  addGrab(descriptor: GrabDescriptor): GrabId {
    const body = this.requireBody(descriptor.body)
    const position = readVec3(descriptor.position)
    const radiusSquared = descriptor.radius * descriptor.radius
    let count = 0
    for (let i = 0; i < body.positions.length; i += 3) {
      const dx = body.positions[i] - position[0]
      const dy = body.positions[i + 1] - position[1]
      const dz = body.positions[i + 2] - position[2]
      if (dx * dx + dy * dy + dz * dz <= radiusSquared)
        count++
    }
    const particles = new Uint32Array(count)
    const offsets = new Float32Array(count * 3)
    let item = 0
    for (let i = 0; i < body.positions.length; i += 3) {
      const dx = body.positions[i] - position[0]
      const dy = body.positions[i + 1] - position[1]
      const dz = body.positions[i + 2] - position[2]
      if (dx * dx + dy * dy + dz * dz > radiusSquared)
        continue
      particles[item] = i / 3
      offsets[item * 3] = dx
      offsets[item * 3 + 1] = dy
      offsets[item * 3 + 2] = dz
      item++
    }
    const id = this.nextGrabId++
    this.grabs.set(id, {
      body: descriptor.body,
      compliance: descriptor.compliance ?? 1e-8,
      id,
      offsets,
      particles,
      position,
    })
    return id
  }

  dispose(): void {
    this.bodies.clear()
    this.colliders.clear()
    this.grabs.clear()
    this.integrationKernel?.dispose?.()
  }

  getPositions(id: BodyId): Float32Array {
    return this.requireBody(id).positions
  }

  removeBody(id: BodyId): void {
    this.bodies.delete(id)
    for (const [grabId, grab] of this.grabs) {
      if (grab.body === id)
        this.grabs.delete(grabId)
    }
  }

  removeCollider(id: ColliderId): void {
    this.colliders.delete(id)
  }

  removeGrab(id: GrabId): void {
    this.grabs.delete(id)
  }

  resetBody(id: BodyId, positions?: Float32Array): void {
    const body = this.requireBody(id)
    const source = positions ?? body.initial
    if (source.length !== body.positions.length)
      throw new RangeError('Reset positions do not match the body particle count')
    body.positions.set(source)
    body.previous.set(source)
  }

  setParticleTargets(id: BodyId, indices: Uint32Array, positions: Float32Array): void {
    const body = this.requireBody(id)
    if (positions.length !== indices.length * 3)
      throw new RangeError('Particle target positions must match target indices')
    for (const index of indices) {
      if (index >= body.inverseMasses.length)
        throw new RangeError(`Particle target ${index} is outside body ${id}`)
    }
    if (body.targets?.indices.length === indices.length) {
      body.targets.indices.set(indices)
      body.targets.positions.set(positions)
    }
    else {
      body.targets = { indices: indices.slice(), positions: positions.slice() }
    }
  }

  // eslint-disable-next-line sonarjs/cognitive-complexity
  async step(delta: number, options: BackendStepOptions): Promise<void> {
    if (delta <= 0 || !Number.isFinite(delta))
      throw new RangeError('Step delta must be a positive finite number')
    const substeps = Math.max(1, Math.floor(options.quality.substeps))
    const subDelta = delta / substeps
    const collisionInterval = Math.max(1, Math.floor(options.quality.collisionEverySubsteps))
    this.elapsed += delta

    for (let substep = 0; substep < substeps; substep++) {
      for (const body of this.bodies.values()) {
        this.applyTargets(body)
        const prediction = this.predict(body, subDelta, options.gravity, options.forceFields)
        if (prediction != null)
          await prediction
        this.limitSelfCollisionSpeed(body, subDelta, collisionInterval, substeps, options.speedLimit)
        this.solveDistanceConstraints(body, subDelta)
        this.solveAreaConstraints(body, subDelta)
        this.solveGrabs(body, subDelta)
      }

      // Always leave the rendered frame with fresh contacts, even when a
      // custom collision interval does not divide the substep count.
      const shouldCollide = (substep + 1) % collisionInterval === 0 || substep === substeps - 1
      if (shouldCollide) {
        for (let iteration = 0; iteration < options.quality.collisionIterations; iteration++) {
          this.solveBodyCollisions(options.quality.maxCollisionCandidates)
          this.solveEdgeCollisions(options.quality.maxCollisionCandidates)
          for (const body of this.bodies.values())
            this.solveExternalColliders(body, options.quality.continuousCollision)
        }
      }
      for (const body of this.bodies.values()) {
        this.applyTargets(body)
        this.applyLaplacianDamping(body, subDelta)
      }
    }
  }

  updateCollider(id: ColliderId, descriptor: ColliderDescriptor): void {
    const collider = this.colliders.get(id)
    if (collider == null)
      throw new RangeError(`Unknown collider ${id}`)
    collider.descriptor = descriptor
    collider.filter = mergeFilter(descriptor.collisionFilter)
  }

  updateGrab(id: GrabId, position: Vec3Like): void {
    const grab = this.grabs.get(id)
    if (grab == null)
      throw new RangeError(`Unknown grab ${id}`)
    grab.position = readVec3(position)
  }

  private applyAerodynamics(body: BodyState, delta: number, forceFields: readonly ForceField[]): void {
    const winds = forceFields.filter((field): field is Extract<ForceField, { type: 'wind' }> => field.type === 'wind')
    if (winds.length === 0)
      return
    const wind = winds.reduce<Vec3>((sum, field) => {
      const velocity = readVec3(field.velocity)
      return [sum[0] + velocity[0], sum[1] + velocity[1], sum[2] + velocity[2]]
    }, [0, 0, 0])
    for (let offset = 0; offset < body.indices.length; offset += 3) {
      const a = body.indices[offset]
      const b = body.indices[offset + 1]
      const c = body.indices[offset + 2]
      const ai = a * 3
      const bi = b * 3
      const ci = c * 3
      const abx = body.positions[bi] - body.positions[ai]
      const aby = body.positions[bi + 1] - body.positions[ai + 1]
      const abz = body.positions[bi + 2] - body.positions[ai + 2]
      const acx = body.positions[ci] - body.positions[ai]
      const acy = body.positions[ci + 1] - body.positions[ai + 1]
      const acz = body.positions[ci + 2] - body.positions[ai + 2]
      const crossX = aby * acz - abz * acy
      const crossY = abz * acx - abx * acz
      const crossZ = abx * acy - aby * acx
      const doubleArea = Math.hypot(crossX, crossY, crossZ)
      if (doubleArea < EPSILON)
        continue
      const nx = crossX / doubleArea
      const ny = crossY / doubleArea
      const nz = crossZ / doubleArea
      const vx = ((body.positions[ai] - body.previous[ai]) + (body.positions[bi] - body.previous[bi]) + (body.positions[ci] - body.previous[ci])) / (3 * delta)
      const vy = ((body.positions[ai + 1] - body.previous[ai + 1]) + (body.positions[bi + 1] - body.previous[bi + 1]) + (body.positions[ci + 1] - body.previous[ci + 1])) / (3 * delta)
      const vz = ((body.positions[ai + 2] - body.previous[ai + 2]) + (body.positions[bi + 2] - body.previous[bi + 2]) + (body.positions[ci + 2] - body.previous[ci + 2])) / (3 * delta)
      const relativeX = wind[0] - vx
      const relativeY = wind[1] - vy
      const relativeZ = wind[2] - vz
      const normalSpeed = relativeX * nx + relativeY * ny + relativeZ * nz
      const material = materialForTriangle(body, offset / 3)
      const dragScale = normalSpeed * Math.abs(normalSpeed) * doubleArea * 0.5 * material.drag * delta * delta / 3
      const liftScale = Math.abs(normalSpeed) * doubleArea * 0.5 * material.lift * delta * delta / 3
      for (const particle of [a, b, c]) {
        if (body.inverseMasses[particle] === 0)
          continue
        const particleOffset = particle * 3
        body.positions[particleOffset] += (nx * dragScale + relativeX * liftScale) * body.inverseMasses[particle]
        body.positions[particleOffset + 1] += (ny * dragScale + relativeY * liftScale) * body.inverseMasses[particle]
        body.positions[particleOffset + 2] += (nz * dragScale + relativeZ * liftScale) * body.inverseMasses[particle]
      }
    }
  }

  private applyContactCorrection(
    body: BodyState,
    particle: number,
    nx: number,
    ny: number,
    nz: number,
    correction: number,
  ): void {
    const offset = particle * 3
    const dx = nx * correction
    const dy = ny * correction
    const dz = nz * correction
    body.positions[offset] += dx
    body.positions[offset + 1] += dy
    body.positions[offset + 2] += dz
    // Position projection must not turn initial overlap into explosive velocity.
    body.previous[offset] += dx
    body.previous[offset + 1] += dy
    body.previous[offset + 2] += dz
    const vx = body.positions[offset] - body.previous[offset]
    const vy = body.positions[offset + 1] - body.previous[offset + 1]
    const vz = body.positions[offset + 2] - body.previous[offset + 2]
    const inwardVelocity = vx * nx + vy * ny + vz * nz
    if (inwardVelocity < 0) {
      body.previous[offset] += nx * inwardVelocity
      body.previous[offset + 1] += ny * inwardVelocity
      body.previous[offset + 2] += nz * inwardVelocity
    }
  }

  private applyFriction(body: BodyState, particle: number, nx: number, ny: number, nz: number, coefficient: number): void {
    const offset = particle * 3
    const vx = body.positions[offset] - body.previous[offset]
    const vy = body.positions[offset + 1] - body.previous[offset + 1]
    const vz = body.positions[offset + 2] - body.previous[offset + 2]
    const normalVelocity = vx * nx + vy * ny + vz * nz
    const tx = vx - nx * normalVelocity
    const ty = vy - ny * normalVelocity
    const tz = vz - nz * normalVelocity
    const friction = Math.max(0, Math.min(1, coefficient))
    body.previous[offset] += tx * friction
    body.previous[offset + 1] += ty * friction
    body.previous[offset + 2] += tz * friction
  }

  private applyLaplacianDamping(body: BodyState, delta: number): void {
    const materialAmount = Math.max(0, Math.min(1, body.materials[0].laplacianDamping))
    if (materialAmount <= 0)
      return
    // Material damping is authored as a 60 Hz amount. Converting it to a
    // substep amount keeps the effect stable when quality changes substep count.
    const amount = 1 - (1 - materialAmount) ** (delta * 60)
    const velocities = body.velocityScratch
    for (let particle = 0; particle < body.inverseMasses.length; particle++) {
      const offset = particle * 3
      velocities[offset] = body.positions[offset] - body.previous[offset]
      velocities[offset + 1] = body.positions[offset + 1] - body.previous[offset + 1]
      velocities[offset + 2] = body.positions[offset + 2] - body.previous[offset + 2]
    }
    const { adjacency, adjacencyOffsets } = body.topology
    for (let particle = 0; particle < body.inverseMasses.length; particle++) {
      if (body.inverseMasses[particle] === 0)
        continue
      const start = adjacencyOffsets[particle]
      const end = adjacencyOffsets[particle + 1]
      const count = end - start
      if (count === 0)
        continue
      let averageX = 0
      let averageY = 0
      let averageZ = 0
      for (let index = start; index < end; index++) {
        const neighborOffset = adjacency[index] * 3
        averageX += velocities[neighborOffset]
        averageY += velocities[neighborOffset + 1]
        averageZ += velocities[neighborOffset + 2]
      }
      const offset = particle * 3
      const velocityX = velocities[offset] + (averageX / count - velocities[offset]) * amount
      const velocityY = velocities[offset + 1] + (averageY / count - velocities[offset + 1]) * amount
      const velocityZ = velocities[offset + 2] + (averageZ / count - velocities[offset + 2]) * amount
      body.previous[offset] = body.positions[offset] - velocityX
      body.previous[offset + 1] = body.positions[offset + 1] - velocityY
      body.previous[offset + 2] = body.positions[offset + 2] - velocityZ
    }
  }

  private applyTargets(body: BodyState): void {
    const targets = body.targets
    if (targets == null)
      return
    for (let i = 0; i < targets.indices.length; i++) {
      const particle = targets.indices[i]
      const offset = particle * 3
      const targetOffset = i * 3
      body.positions[offset] = targets.positions[targetOffset]!
      body.positions[offset + 1] = targets.positions[targetOffset + 1]!
      body.positions[offset + 2] = targets.positions[targetOffset + 2]!
      body.previous[offset] = targets.positions[targetOffset]!
      body.previous[offset + 1] = targets.positions[targetOffset + 1]!
      body.previous[offset + 2] = targets.positions[targetOffset + 2]!
    }
  }

  private limitSelfCollisionSpeed(
    body: BodyState,
    delta: number,
    collisionInterval: number,
    substepCount: number,
    speedLimit: ClothSpeedLimit,
  ): void {
    if (!body.selfCollision || speedLimit === 'unlimited') {
      body.maximumSelfCollisionDepenetration = Number.POSITIVE_INFINITY
      body.maximumSelfCollisionDisplacement = Number.POSITIVE_INFINITY
      return
    }
    const localScale = body.particleSpacing > EPSILON
      ? Math.min(body.collisionThickness * 2, body.particleSpacing * 0.5)
      : body.collisionThickness * 2
    const depenetrationScale = body.particleSpacing > EPSILON
      ? Math.min(body.collisionThickness, body.particleSpacing * 0.5)
      : body.collisionThickness
    const maximumDisplacement = typeof speedLimit === 'number'
      ? speedLimit * delta
      : Math.max(EPSILON, localScale) / (collisionInterval * 2)
    body.maximumSelfCollisionDepenetration = typeof speedLimit === 'number'
      ? maximumDisplacement
      : Math.max(EPSILON, depenetrationScale) / substepCount
    body.maximumSelfCollisionDisplacement = maximumDisplacement
    const maximumSquared = maximumDisplacement * maximumDisplacement
    for (let particle = 0; particle < body.inverseMasses.length; particle++) {
      if (body.inverseMasses[particle] === 0)
        continue
      const offset = particle * 3
      const dx = body.positions[offset] - body.previous[offset]
      const dy = body.positions[offset + 1] - body.previous[offset + 1]
      const dz = body.positions[offset + 2] - body.previous[offset + 2]
      const distanceSquared = dx * dx + dy * dy + dz * dz
      if (distanceSquared <= maximumSquared)
        continue
      const scale = maximumDisplacement / Math.sqrt(distanceSquared)
      body.positions[offset] = body.previous[offset] + dx * scale
      body.positions[offset + 1] = body.previous[offset + 1] + dy * scale
      body.positions[offset + 2] = body.previous[offset + 2] + dz * scale
    }
  }

  private predict(body: BodyState, delta: number, gravity: Vec3, forceFields: readonly ForceField[]): Promise<void> | void {
    const deltaSquared = delta * delta
    const damping = Math.max(0, Math.min(1, body.materials[0].damping))
    const accelerations = body.accelerations
    for (let i = 0; i < body.inverseMasses.length; i++) {
      const offset = i * 3
      const x = body.positions[offset]
      const y = body.positions[offset + 1]
      const z = body.positions[offset + 2]
      let ax = gravity[0]
      let ay = gravity[1]
      let az = gravity[2]
      for (const field of forceFields) {
        const force = this.sampleForce(field, x, y, z)
        ax += force[0]
        ay += force[1]
        az += force[2]
      }
      accelerations[offset] = ax
      accelerations[offset + 1] = ay
      accelerations[offset + 2] = az
    }
    if (this.integrationKernel != null) {
      const result = this.integrationKernel.integrate(body.positions, body.previous, body.inverseMasses, accelerations, delta, damping)
      if (result != null) {
        return result.then(() => this.applyAerodynamics(body, delta, forceFields))
      }
    }
    else {
      for (let i = 0; i < body.inverseMasses.length; i++) {
        if (body.inverseMasses[i] === 0)
          continue
        const offset = i * 3
        const x = body.positions[offset]
        const y = body.positions[offset + 1]
        const z = body.positions[offset + 2]
        const vx = (x - body.previous[offset]) * (1 - damping)
        const vy = (y - body.previous[offset + 1]) * (1 - damping)
        const vz = (z - body.previous[offset + 2]) * (1 - damping)
        body.previous[offset] = x
        body.previous[offset + 1] = y
        body.previous[offset + 2] = z
        body.positions[offset] = x + vx + accelerations[offset] * deltaSquared
        body.positions[offset + 1] = y + vy + accelerations[offset + 1] * deltaSquared
        body.positions[offset + 2] = z + vz + accelerations[offset + 2] * deltaSquared
      }
    }
    this.applyAerodynamics(body, delta, forceFields)
  }

  private requireBody(id: BodyId): BodyState {
    const body = this.bodies.get(id)
    if (body == null)
      throw new RangeError(`Unknown cloth body ${id}`)
    return body
  }

  private sampleForce(field: ForceField, x: number, y: number, z: number): Vec3 {
    if (field.type === 'directional')
      return readVec3(field.acceleration)
    if (field.type === 'wind')
      return [0, 0, 0]
    if (field.type === 'noise') {
      const frequency = field.frequency
      const seed = field.seed ?? 0
      return [
        hashNoise(x * frequency + this.elapsed, y, z, seed) * field.amplitude,
        hashNoise(x, y * frequency + this.elapsed, z, seed + 1) * field.amplitude,
        hashNoise(x, y, z * frequency + this.elapsed, seed + 2) * field.amplitude,
      ]
    }
    const source = readVec3(field.type === 'point' ? field.position : field.origin)
    const dx = x - source[0]
    const dy = y - source[1]
    const dz = z - source[2]
    const distance = Math.hypot(dx, dy, dz)
    if (distance >= field.radius || distance < EPSILON)
      return [0, 0, 0]
    const scale = (1 - distance / field.radius) ** ('falloff' in field ? (field.falloff ?? 2) : 1)
    if (field.type === 'point')
      return [-dx / distance * field.strength * scale, -dy / distance * field.strength * scale, -dz / distance * field.strength * scale]
    const [axisX, axisY, axisZ] = normalize(...readVec3(field.axis))
    const tangent = normalize(axisY * dz - axisZ * dy, axisZ * dx - axisX * dz, axisX * dy - axisY * dx)
    return [tangent[0] * field.strength * scale, tangent[1] * field.strength * scale, tangent[2] * field.strength * scale]
  }

  private solveAreaConstraints(body: BodyState, delta: number): void {
    const alpha = body.materials[0].shearCompliance / (delta * delta)
    const particles = body.topology.triangleParticles
    for (let triangle = 0; triangle < body.topology.triangleRestAreas.length; triangle++) {
      const particleOffset = triangle * 3
      const a = particles[particleOffset]
      const b = particles[particleOffset + 1]
      const c = particles[particleOffset + 2]
      const ai = a * 3
      const bi = b * 3
      const ci = c * 3
      const ax = body.positions[ai]
      const ay = body.positions[ai + 1]
      const az = body.positions[ai + 2]
      const bx = body.positions[bi]
      const by = body.positions[bi + 1]
      const bz = body.positions[bi + 2]
      const cx = body.positions[ci]
      const cy = body.positions[ci + 1]
      const cz = body.positions[ci + 2]
      const crossX = (by - ay) * (cz - az) - (bz - az) * (cy - ay)
      const crossY = (bz - az) * (cx - ax) - (bx - ax) * (cz - az)
      const crossZ = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
      const doubleArea = Math.hypot(crossX, crossY, crossZ)
      if (doubleArea < EPSILON)
        continue
      const nx = crossX / doubleArea
      const ny = crossY / doubleArea
      const nz = crossZ / doubleArea
      const gax = (by - cy) * nz - (bz - cz) * ny
      const gay = (bz - cz) * nx - (bx - cx) * nz
      const gaz = (bx - cx) * ny - (by - cy) * nx
      const gbx = (cy - ay) * nz - (cz - az) * ny
      const gby = (cz - az) * nx - (cx - ax) * nz
      const gbz = (cx - ax) * ny - (cy - ay) * nx
      const gcx = (ay - by) * nz - (az - bz) * ny
      const gcy = (az - bz) * nx - (ax - bx) * nz
      const gcz = (ax - bx) * ny - (ay - by) * nx
      const wa = body.inverseMasses[a]
      const wb = body.inverseMasses[b]
      const wc = body.inverseMasses[c]
      const denominator = alpha + 0.25 * (
        wa * (gax * gax + gay * gay + gaz * gaz)
        + wb * (gbx * gbx + gby * gby + gbz * gbz)
        + wc * (gcx * gcx + gcy * gcy + gcz * gcz)
      )
      if (denominator < EPSILON)
        continue
      const lambda = -(doubleArea * 0.5 - body.topology.triangleRestAreas[triangle]) / denominator
      const scaleA = lambda * wa * 0.5
      const scaleB = lambda * wb * 0.5
      const scaleC = lambda * wc * 0.5
      body.positions[ai] += gax * scaleA
      body.positions[ai + 1] += gay * scaleA
      body.positions[ai + 2] += gaz * scaleA
      body.positions[bi] += gbx * scaleB
      body.positions[bi + 1] += gby * scaleB
      body.positions[bi + 2] += gbz * scaleB
      body.positions[ci] += gcx * scaleC
      body.positions[ci + 1] += gcy * scaleC
      body.positions[ci + 2] += gcz * scaleC
    }
  }

  // eslint-disable-next-line sonarjs/cognitive-complexity
  private solveBodyCollisions(maxCandidates: number): void {
    const bodies = [...this.bodies.values()]
    if (bodies.length === 0)
      return
    const maxThickness = Math.max(...bodies.flatMap(body => body.materials.map(material => material.thickness)))
    const inverseCellSize = 1 / Math.max(...bodies.map(body => body.collisionCellSize))
    const cells = new Map<number, number[]>()
    const candidates: { body: BodyState, offset: number }[] = []

    for (const body of bodies) {
      for (let offset = 0; offset < body.indices.length; offset += 3) {
        const candidate = candidates.length
        candidates.push({ body, offset })
        const a = body.indices[offset] * 3
        const b = body.indices[offset + 1] * 3
        const c = body.indices[offset + 2] * 3
        const minX = Math.min(body.positions[a], body.positions[b], body.positions[c]) - maxThickness
        const minY = Math.min(body.positions[a + 1], body.positions[b + 1], body.positions[c + 1]) - maxThickness
        const minZ = Math.min(body.positions[a + 2], body.positions[b + 2], body.positions[c + 2]) - maxThickness
        const maxX = Math.max(body.positions[a], body.positions[b], body.positions[c]) + maxThickness
        const maxY = Math.max(body.positions[a + 1], body.positions[b + 1], body.positions[c + 1]) + maxThickness
        const maxZ = Math.max(body.positions[a + 2], body.positions[b + 2], body.positions[c + 2]) + maxThickness
        const fromX = Math.floor(minX * inverseCellSize)
        const fromY = Math.floor(minY * inverseCellSize)
        const fromZ = Math.floor(minZ * inverseCellSize)
        const toX = Math.ceil(maxX * inverseCellSize)
        const toY = Math.ceil(maxY * inverseCellSize)
        const toZ = Math.ceil(maxZ * inverseCellSize)
        let inserted = 0
        for (let x = fromX; x <= toX && inserted < 64; x++) {
          for (let y = fromY; y <= toY && inserted < 64; y++) {
            for (let z = fromZ; z <= toZ && inserted < 64; z++) {
              const key = ((x * 73_856_093) ^ (y * 19_349_663) ^ (z * 83_492_791)) >>> 0
              const cell = cells.get(key)
              if (cell == null)
                cells.set(key, [candidate])
              else
                cell.push(candidate)
              inserted++
            }
          }
        }
      }
    }

    for (const vertexBody of bodies) {
      for (let particle = 0; particle < vertexBody.inverseMasses.length; particle++) {
        if (vertexBody.inverseMasses[particle] === 0)
          continue
        const particleOffset = particle * 3
        const cell = cells.get(cellKey(
          vertexBody.positions[particleOffset],
          vertexBody.positions[particleOffset + 1],
          vertexBody.positions[particleOffset + 2],
          inverseCellSize,
        ))
        if (cell == null)
          continue
        let tested = 0
        for (const candidateIndex of cell) {
          if (tested >= maxCandidates)
            break
          const candidate = candidates[candidateIndex]
          const triangleBody = candidate.body
          if (triangleBody === vertexBody && !vertexBody.selfCollision)
            continue
          if (triangleBody !== vertexBody && !filtersCollide(vertexBody.filter, triangleBody.filter))
            continue
          const a = triangleBody.indices[candidate.offset]
          const b = triangleBody.indices[candidate.offset + 1]
          const c = triangleBody.indices[candidate.offset + 2]
          if (triangleBody === vertexBody && (
            particle === a || particle === b || particle === c
            || isWithinTwoRings(vertexBody.topology, particle, a)
            || isWithinTwoRings(vertexBody.topology, particle, b)
            || isWithinTwoRings(vertexBody.topology, particle, c)
          )) {
            continue
          }
          tested++
          this.solveVertexTriangle(vertexBody, particle, triangleBody, candidate.offset)
        }
      }
    }
  }

  private solveCollider(body: BodyState, particle: number, descriptor: ColliderDescriptor): void {
    const shape = descriptor.shape
    const offset = particle * 3
    const position: Vec3 = [body.positions[offset], body.positions[offset + 1], body.positions[offset + 2]]
    const thickness = body.materials[0].thickness
    let closest: Vec3
    let targetDistance = thickness
    if (shape.type === 'sphere') {
      closest = readVec3(shape.center)
      targetDistance += shape.radius
    }
    else if (shape.type === 'capsule') {
      const point = closestPointOnSegment(...position, ...readVec3(shape.start), ...readVec3(shape.end))
      closest = [point[0], point[1], point[2]]
      targetDistance += shape.radius
    }
    else if (shape.type === 'plane') {
      const normal = normalize(...readVec3(shape.normal))
      const constant = shape.constant ?? -(shape.offset ?? 0)
      const signed = position[0] * normal[0] + position[1] * normal[1] + position[2] * normal[2] + constant
      if (signed >= thickness)
        return
      const correction = thickness - signed
      this.applyContactCorrection(body, particle, ...normal, correction)
      this.applyFriction(body, particle, ...normal, descriptor.friction ?? body.materials[0].kineticFriction)
      return
    }
    else if (shape.type === 'roundedBox') {
      const center = readVec3(shape.center)
      const halfExtents = readVec3(shape.halfExtents)
      const rotation = shape.rotation == null ? undefined : readQuat(shape.rotation)
      const translated: Vec3 = [position[0] - center[0], position[1] - center[1], position[2] - center[2]]
      const local = rotation == null ? translated : inverseRotate(translated, rotation)
      const boxPoint: Vec3 = [
        Math.max(-halfExtents[0], Math.min(halfExtents[0], local[0])),
        Math.max(-halfExtents[1], Math.min(halfExtents[1], local[1])),
        Math.max(-halfExtents[2], Math.min(halfExtents[2], local[2])),
      ]
      const worldPoint = rotation == null ? boxPoint : rotate(boxPoint, rotation)
      closest = [worldPoint[0] + center[0], worldPoint[1] + center[1], worldPoint[2] + center[2]]
      targetDistance += shape.radius
    }
    else {
      this.solveTriangleMeshCollider(body, particle, descriptor)
      return
    }

    const dx = position[0] - closest[0]
    const dy = position[1] - closest[1]
    const dz = position[2] - closest[2]
    const length = Math.hypot(dx, dy, dz)
    if (length >= targetDistance)
      return
    const normal = normalize(dx, dy, dz)
    const correction = targetDistance - length
    this.applyContactCorrection(body, particle, ...normal, correction)
    this.applyFriction(body, particle, ...normal, descriptor.friction ?? body.materials[0].kineticFriction)
  }

  private solveDistanceConstraints(body: BodyState, delta: number): void {
    const topology = body.topology
    for (let constraint = 0; constraint < topology.edgeKinds.length; constraint++) {
      const a = topology.edgeParticles[constraint * 2]
      const b = topology.edgeParticles[constraint * 2 + 1]
      const ai = a * 3
      const bi = b * 3
      const dx = body.positions[ai] - body.positions[bi]
      const dy = body.positions[ai + 1] - body.positions[bi + 1]
      const dz = body.positions[ai + 2] - body.positions[bi + 2]
      const length = Math.hypot(dx, dy, dz)
      if (length < EPSILON)
        continue
      const wa = body.inverseMasses[a]
      const wb = body.inverseMasses[b]
      const compliance = topology.edgeKinds[constraint] === STRETCH_EDGE
        ? body.materials[0].stretchCompliance
        : body.materials[0].bendCompliance
      const alpha = compliance / (delta * delta)
      const lambda = -(length - topology.edgeRestLengths[constraint]) / (wa + wb + alpha)
      const scale = lambda / length
      body.positions[ai] += dx * scale * wa
      body.positions[ai + 1] += dy * scale * wa
      body.positions[ai + 2] += dz * scale * wa
      body.positions[bi] -= dx * scale * wb
      body.positions[bi + 1] -= dy * scale * wb
      body.positions[bi + 2] -= dz * scale * wb
    }
  }

  // eslint-disable-next-line sonarjs/cognitive-complexity
  private solveEdgeCollisions(maxCandidates: number): void {
    const bodies = [...this.bodies.values()]
    const entries: { a: number, b: number, body: BodyState }[] = []
    const cells = new Map<number, number[]>()
    const thickness = Math.max(0.01, ...bodies.map(body => body.collisionCellSize))
    const inverseCellSize = 1 / thickness
    for (const body of bodies) {
      const topology = body.topology
      for (let edge = 0; edge < topology.edgeKinds.length; edge++) {
        if (topology.edgeKinds[edge] !== STRETCH_EDGE)
          continue
        const a = topology.edgeParticles[edge * 2]
        const b = topology.edgeParticles[edge * 2 + 1]
        const ai = a * 3
        const bi = b * 3
        const index = entries.length
        entries.push({ a, b, body })
        const key = cellKey(
          (body.positions[ai] + body.positions[bi]) * 0.5,
          (body.positions[ai + 1] + body.positions[bi + 1]) * 0.5,
          (body.positions[ai + 2] + body.positions[bi + 2]) * 0.5,
          inverseCellSize,
        )
        const list = cells.get(key) ?? []
        list.push(index)
        cells.set(key, list)
      }
    }
    for (const list of cells.values()) {
      let tested = 0
      for (let i = 0; i < list.length; i++) {
        for (let j = i + 1; j < list.length && tested < maxCandidates * list.length; j++, tested++) {
          const first = entries[list[i]]
          const second = entries[list[j]]
          if (first.body === second.body) {
            if (
              !first.body.selfCollision
              || isWithinTwoRings(first.body.topology, first.a, second.a)
              || isWithinTwoRings(first.body.topology, first.a, second.b)
              || isWithinTwoRings(first.body.topology, first.b, second.a)
              || isWithinTwoRings(first.body.topology, first.b, second.b)
            ) {
              continue
            }
          }
          else if (!filtersCollide(first.body.filter, second.body.filter)) {
            continue
          }
          this.solveEdgeEdge(first.body, first.a, first.b, second.body, second.a, second.b)
        }
      }
    }
  }

  // eslint-disable-next-line sonarjs/cognitive-complexity
  private solveEdgeEdge(firstBody: BodyState, a: number, b: number, secondBody: BodyState, c: number, d: number): void {
    const ai = a * 3
    const bi = b * 3
    const ci = c * 3
    const di = d * 3
    const closest = closestSegmentSegment(
      [firstBody.positions[ai], firstBody.positions[ai + 1], firstBody.positions[ai + 2]],
      [firstBody.positions[bi], firstBody.positions[bi + 1], firstBody.positions[bi + 2]],
      [secondBody.positions[ci], secondBody.positions[ci + 1], secondBody.positions[ci + 2]],
      [secondBody.positions[di], secondBody.positions[di + 1], secondBody.positions[di + 2]],
    )
    let dx = closest[0] - closest[3]
    let dy = closest[1] - closest[4]
    let dz = closest[2] - closest[5]
    const length = Math.hypot(dx, dy, dz)
    const target = firstBody.materials[0].thickness + secondBody.materials[0].thickness
    if (length >= target) {
      return
    }
    const previousClosest = closestSegmentSegment(
      [firstBody.previous[ai], firstBody.previous[ai + 1], firstBody.previous[ai + 2]],
      [firstBody.previous[bi], firstBody.previous[bi + 1], firstBody.previous[bi + 2]],
      [secondBody.previous[ci], secondBody.previous[ci + 1], secondBody.previous[ci + 2]],
      [secondBody.previous[di], secondBody.previous[di + 1], secondBody.previous[di + 2]],
    )
    const previousDx = previousClosest[0] - previousClosest[3]
    const previousDy = previousClosest[1] - previousClosest[4]
    const previousDz = previousClosest[2] - previousClosest[5]
    const previousLength = Math.hypot(previousDx, previousDy, previousDz)
    let initialPenetration = 0
    let restDx = 0
    let restDy = 0
    let restDz = 0
    if (firstBody === secondBody) {
      const restClosest = closestSegmentSegment(
        [firstBody.initial[ai], firstBody.initial[ai + 1], firstBody.initial[ai + 2]],
        [firstBody.initial[bi], firstBody.initial[bi + 1], firstBody.initial[bi + 2]],
        [secondBody.initial[ci], secondBody.initial[ci + 1], secondBody.initial[ci + 2]],
        [secondBody.initial[di], secondBody.initial[di + 1], secondBody.initial[di + 2]],
      )
      restDx = restClosest[0] - restClosest[3]
      restDy = restClosest[1] - restClosest[4]
      restDz = restClosest[2] - restClosest[5]
      const restDistance = Math.hypot(restDx, restDy, restDz)
      if (restDistance < target * 1.25)
        return
      initialPenetration = Math.max(0, target - previousLength)
    }
    if (length >= EPSILON) {
      dx /= length
      dy /= length
      dz /= length
      // Preserve the previous side when two edges cross during a discrete
      // step; otherwise the unsigned closest-point normal can flip.
      if (previousLength >= EPSILON && dx * previousDx + dy * previousDy + dz * previousDz < 0) {
        dx = -dx
        dy = -dy
        dz = -dz
      }
    }
    else if (previousLength >= EPSILON) {
      dx = previousDx / previousLength
      dy = previousDy / previousLength
      dz = previousDz / previousLength
    }
    else {
      const restLength = Math.hypot(restDx, restDy, restDz)
      if (restLength >= EPSILON) {
        dx = restDx / restLength
        dy = restDy / restLength
        dz = restDz / restLength
      }
      else {
        const firstX = firstBody.positions[bi] - firstBody.positions[ai]
        const firstY = firstBody.positions[bi + 1] - firstBody.positions[ai + 1]
        const firstZ = firstBody.positions[bi + 2] - firstBody.positions[ai + 2]
        const secondX = secondBody.positions[di] - secondBody.positions[ci]
        const secondY = secondBody.positions[di + 1] - secondBody.positions[ci + 1]
        const secondZ = secondBody.positions[di + 2] - secondBody.positions[ci + 2]
        ;[dx, dy, dz] = normalize(
          firstY * secondZ - firstZ * secondY,
          firstZ * secondX - firstX * secondZ,
          firstX * secondY - firstY * secondX,
        )
      }
    }
    const weights = [
      firstBody.inverseMasses[a] * (1 - closest[6]) ** 2,
      firstBody.inverseMasses[b] * closest[6] ** 2,
      secondBody.inverseMasses[c] * (1 - closest[7]) ** 2,
      secondBody.inverseMasses[d] * closest[7] ** 2,
    ] as const
    const denominator = weights.reduce((sum, weight) => sum + weight, 0)
    if (denominator < EPSILON)
      return
    const normalCorrection = limitedDepenetration(
      target - length,
      initialPenetration,
      firstBody === secondBody ? firstBody.maximumSelfCollisionDepenetration : Number.POSITIVE_INFINITY,
    )
    const correction = normalCorrection / denominator
    const particles = [[firstBody, a, 1 - closest[6], 1], [firstBody, b, closest[6], 1], [secondBody, c, 1 - closest[7], -1], [secondBody, d, closest[7], -1]] as const
    for (const [body, particle, interpolation, direction] of particles) {
      const offset = particle * 3
      const scale = correction * body.inverseMasses[particle] * interpolation * direction
      const correctionX = dx * scale
      const correctionY = dy * scale
      const correctionZ = dz * scale
      body.positions[offset] += correctionX
      body.positions[offset + 1] += correctionY
      body.positions[offset + 2] += correctionZ
      body.previous[offset] += correctionX
      body.previous[offset + 1] += correctionY
      body.previous[offset + 2] += correctionZ
    }

    const velocityA = [
      firstBody.positions[ai] - firstBody.previous[ai],
      firstBody.positions[ai + 1] - firstBody.previous[ai + 1],
      firstBody.positions[ai + 2] - firstBody.previous[ai + 2],
    ] as const
    const velocityB = [
      firstBody.positions[bi] - firstBody.previous[bi],
      firstBody.positions[bi + 1] - firstBody.previous[bi + 1],
      firstBody.positions[bi + 2] - firstBody.previous[bi + 2],
    ] as const
    const velocityC = [
      secondBody.positions[ci] - secondBody.previous[ci],
      secondBody.positions[ci + 1] - secondBody.previous[ci + 1],
      secondBody.positions[ci + 2] - secondBody.previous[ci + 2],
    ] as const
    const velocityD = [
      secondBody.positions[di] - secondBody.previous[di],
      secondBody.positions[di + 1] - secondBody.previous[di + 1],
      secondBody.positions[di + 2] - secondBody.previous[di + 2],
    ] as const
    const relativeX = velocityA[0] * (1 - closest[6]) + velocityB[0] * closest[6] - velocityC[0] * (1 - closest[7]) - velocityD[0] * closest[7]
    const relativeY = velocityA[1] * (1 - closest[6]) + velocityB[1] * closest[6] - velocityC[1] * (1 - closest[7]) - velocityD[1] * closest[7]
    const relativeZ = velocityA[2] * (1 - closest[6]) + velocityB[2] * closest[6] - velocityC[2] * (1 - closest[7]) - velocityD[2] * closest[7]
    const normalVelocity = relativeX * dx + relativeY * dy + relativeZ * dz
    const tangentX = relativeX - dx * normalVelocity
    const tangentY = relativeY - dy * normalVelocity
    const tangentZ = relativeZ - dz * normalVelocity
    const tangentLength = Math.hypot(tangentX, tangentY, tangentZ)
    const friction = Math.max(0, (firstBody.materials[0].kineticFriction + secondBody.materials[0].kineticFriction) * 0.5)
    const tangentScale = tangentLength < EPSILON ? 0 : Math.min(1, friction * normalCorrection / tangentLength)
    const velocityCorrectionX = (dx * Math.max(0, -normalVelocity) - tangentX * tangentScale) / denominator
    const velocityCorrectionY = (dy * Math.max(0, -normalVelocity) - tangentY * tangentScale) / denominator
    const velocityCorrectionZ = (dz * Math.max(0, -normalVelocity) - tangentZ * tangentScale) / denominator
    for (const [body, particle, interpolation, direction] of particles) {
      const offset = particle * 3
      const scale = body.inverseMasses[particle] * interpolation * direction
      body.previous[offset] -= velocityCorrectionX * scale
      body.previous[offset + 1] -= velocityCorrectionY * scale
      body.previous[offset + 2] -= velocityCorrectionZ * scale
    }
  }

  private solveExternalColliders(body: BodyState, continuous: boolean): void {
    for (const collider of this.colliders.values()) {
      if (!filtersCollide(body.filter, collider.filter))
        continue
      for (let particle = 0; particle < body.inverseMasses.length; particle++) {
        if (body.inverseMasses[particle] === 0)
          continue
        if (continuous) {
          const offset = particle * 3
          const currentX = body.positions[offset]
          const currentY = body.positions[offset + 1]
          const currentZ = body.positions[offset + 2]
          const previousX = body.previous[offset]
          const previousY = body.previous[offset + 1]
          const previousZ = body.previous[offset + 2]
          const midpointX = (currentX + previousX) * 0.5
          const midpointY = (currentY + previousY) * 0.5
          const midpointZ = (currentZ + previousZ) * 0.5
          body.positions[offset] = midpointX
          body.positions[offset + 1] = midpointY
          body.positions[offset + 2] = midpointZ
          this.solveCollider(body, particle, collider.descriptor)
          const correctionX = body.positions[offset] - midpointX
          const correctionY = body.positions[offset + 1] - midpointY
          const correctionZ = body.positions[offset + 2] - midpointZ
          body.positions[offset] = currentX + correctionX
          body.positions[offset + 1] = currentY + correctionY
          body.positions[offset + 2] = currentZ + correctionZ
          body.previous[offset] = previousX + correctionX
          body.previous[offset + 1] = previousY + correctionY
          body.previous[offset + 2] = previousZ + correctionZ
        }
        this.solveCollider(body, particle, collider.descriptor)
      }
    }
  }

  private solveGrabs(body: BodyState, delta: number): void {
    for (const grab of this.grabs.values()) {
      if (grab.body !== body.id)
        continue
      const alpha = grab.compliance / (delta * delta)
      for (let i = 0; i < grab.particles.length; i++) {
        const particle = grab.particles[i]
        const offset = particle * 3
        const weight = body.inverseMasses[particle]
        if (weight === 0)
          continue
        const targetX = grab.position[0] + grab.offsets[i * 3]
        const targetY = grab.position[1] + grab.offsets[i * 3 + 1]
        const targetZ = grab.position[2] + grab.offsets[i * 3 + 2]
        const scale = weight / (weight + alpha)
        body.positions[offset] += (targetX - body.positions[offset]) * scale
        body.positions[offset + 1] += (targetY - body.positions[offset + 1]) * scale
        body.positions[offset + 2] += (targetZ - body.positions[offset + 2]) * scale
      }
    }
  }

  private solveTriangleMeshCollider(body: BodyState, particle: number, descriptor: ColliderDescriptor): void {
    const shape = descriptor.shape
    if (shape.type !== 'triangleMesh')
      return
    const offset = particle * 3
    let bestDistance = Number.POSITIVE_INFINITY
    let bestPoint: undefined | Vec3
    for (let i = 0; i < shape.indices.length; i += 3) {
      const a = shape.indices[i] * 3
      const b = shape.indices[i + 1] * 3
      const c = shape.indices[i + 2] * 3
      const point = closestPointOnTriangle(
        body.positions[offset],
        body.positions[offset + 1],
        body.positions[offset + 2],
        shape.positions[a],
        shape.positions[a + 1],
        shape.positions[a + 2],
        shape.positions[b],
        shape.positions[b + 1],
        shape.positions[b + 2],
        shape.positions[c],
        shape.positions[c + 1],
        shape.positions[c + 2],
      )
      const distance = Math.hypot(body.positions[offset] - point[0], body.positions[offset + 1] - point[1], body.positions[offset + 2] - point[2])
      if (distance < bestDistance) {
        bestDistance = distance
        bestPoint = [point[0], point[1], point[2]]
      }
    }
    const thickness = body.materials[0].thickness
    if (bestPoint == null || bestDistance >= thickness)
      return
    const normal = normalize(body.positions[offset] - bestPoint[0], body.positions[offset + 1] - bestPoint[1], body.positions[offset + 2] - bestPoint[2])
    const correction = thickness - bestDistance
    this.applyContactCorrection(body, particle, ...normal, correction)
    this.applyFriction(body, particle, ...normal, descriptor.friction ?? body.materials[0].kineticFriction)
  }

  private solveVertexTriangle(vertexBody: BodyState, particle: number, triangleBody: BodyState, triangleOffset: number): void {
    const p = particle * 3
    const a = triangleBody.indices[triangleOffset]
    const b = triangleBody.indices[triangleOffset + 1]
    const c = triangleBody.indices[triangleOffset + 2]
    const ai = a * 3
    const bi = b * 3
    const ci = c * 3
    const closest = closestPointOnTriangle(
      vertexBody.positions[p],
      vertexBody.positions[p + 1],
      vertexBody.positions[p + 2],
      triangleBody.positions[ai],
      triangleBody.positions[ai + 1],
      triangleBody.positions[ai + 2],
      triangleBody.positions[bi],
      triangleBody.positions[bi + 1],
      triangleBody.positions[bi + 2],
      triangleBody.positions[ci],
      triangleBody.positions[ci + 1],
      triangleBody.positions[ci + 2],
    )
    let dx = vertexBody.positions[p] - closest[0]
    let dy = vertexBody.positions[p + 1] - closest[1]
    let dz = vertexBody.positions[p + 2] - closest[2]
    let length = Math.hypot(dx, dy, dz)
    const actualDistance = length
    const thickness = vertexBody.materials[0].thickness + materialForTriangle(triangleBody, triangleOffset / 3).thickness
    if (length >= thickness)
      return
    let initialPenetration = 0
    if (vertexBody === triangleBody) {
      const restClosest = closestPointOnTriangle(
        vertexBody.initial[p],
        vertexBody.initial[p + 1],
        vertexBody.initial[p + 2],
        triangleBody.initial[ai],
        triangleBody.initial[ai + 1],
        triangleBody.initial[ai + 2],
        triangleBody.initial[bi],
        triangleBody.initial[bi + 1],
        triangleBody.initial[bi + 2],
        triangleBody.initial[ci],
        triangleBody.initial[ci + 1],
        triangleBody.initial[ci + 2],
      )
      const restDistance = Math.hypot(
        vertexBody.initial[p] - restClosest[0],
        vertexBody.initial[p + 1] - restClosest[1],
        vertexBody.initial[p + 2] - restClosest[2],
      )
      // Indexed render meshes often split vertices along UV seams. Treat those
      // coincident rest surfaces as topology, not as an initial collision.
      if (restDistance < thickness * 1.25)
        return
      const previousClosest = closestPointOnTriangle(
        vertexBody.previous[p],
        vertexBody.previous[p + 1],
        vertexBody.previous[p + 2],
        triangleBody.previous[ai],
        triangleBody.previous[ai + 1],
        triangleBody.previous[ai + 2],
        triangleBody.previous[bi],
        triangleBody.previous[bi + 1],
        triangleBody.previous[bi + 2],
        triangleBody.previous[ci],
        triangleBody.previous[ci + 1],
        triangleBody.previous[ci + 2],
      )
      initialPenetration = Math.max(0, thickness - Math.hypot(
        vertexBody.previous[p] - previousClosest[0],
        vertexBody.previous[p + 1] - previousClosest[1],
        vertexBody.previous[p + 2] - previousClosest[2],
      ))
    }
    if (length < EPSILON) {
      const abx = triangleBody.positions[bi] - triangleBody.positions[ai]
      const aby = triangleBody.positions[bi + 1] - triangleBody.positions[ai + 1]
      const abz = triangleBody.positions[bi + 2] - triangleBody.positions[ai + 2]
      const acx = triangleBody.positions[ci] - triangleBody.positions[ai]
      const acy = triangleBody.positions[ci + 1] - triangleBody.positions[ai + 1]
      const acz = triangleBody.positions[ci + 2] - triangleBody.positions[ai + 2]
      ;[dx, dy, dz] = normalize(aby * acz - abz * acy, abz * acx - abx * acz, abx * acy - aby * acx)
      length = 1
    }
    const nx = dx / length
    const ny = dy / length
    const nz = dz / length
    const wp = vertexBody.inverseMasses[particle]
    const wa = triangleBody.inverseMasses[a]
    const wb = triangleBody.inverseMasses[b]
    const wc = triangleBody.inverseMasses[c]
    const denominator = wp + wa * closest[3] ** 2 + wb * closest[4] ** 2 + wc * closest[5] ** 2
    if (denominator < EPSILON)
      return
    const normalCorrection = limitedDepenetration(
      thickness - actualDistance,
      initialPenetration,
      vertexBody === triangleBody ? vertexBody.maximumSelfCollisionDepenetration : Number.POSITIVE_INFINITY,
    )
    const correction = normalCorrection / denominator
    const particleCorrectionX = nx * correction * wp
    const particleCorrectionY = ny * correction * wp
    const particleCorrectionZ = nz * correction * wp
    vertexBody.positions[p] += particleCorrectionX
    vertexBody.positions[p + 1] += particleCorrectionY
    vertexBody.positions[p + 2] += particleCorrectionZ
    vertexBody.previous[p] += particleCorrectionX
    vertexBody.previous[p + 1] += particleCorrectionY
    vertexBody.previous[p + 2] += particleCorrectionZ
    const triangleParticles = [[a, closest[3], wa], [b, closest[4], wb], [c, closest[5], wc]] as const
    for (const [triangleParticle, barycentric, weight] of triangleParticles) {
      const offset = triangleParticle * 3
      const scale = correction * barycentric * weight
      const correctionX = nx * scale
      const correctionY = ny * scale
      const correctionZ = nz * scale
      triangleBody.positions[offset] -= correctionX
      triangleBody.positions[offset + 1] -= correctionY
      triangleBody.positions[offset + 2] -= correctionZ
      triangleBody.previous[offset] -= correctionX
      triangleBody.previous[offset + 1] -= correctionY
      triangleBody.previous[offset + 2] -= correctionZ
    }

    const relativeX = vertexBody.positions[p] - vertexBody.previous[p]
      - (triangleBody.positions[ai] - triangleBody.previous[ai]) * closest[3]
      - (triangleBody.positions[bi] - triangleBody.previous[bi]) * closest[4]
      - (triangleBody.positions[ci] - triangleBody.previous[ci]) * closest[5]
    const relativeY = vertexBody.positions[p + 1] - vertexBody.previous[p + 1]
      - (triangleBody.positions[ai + 1] - triangleBody.previous[ai + 1]) * closest[3]
      - (triangleBody.positions[bi + 1] - triangleBody.previous[bi + 1]) * closest[4]
      - (triangleBody.positions[ci + 1] - triangleBody.previous[ci + 1]) * closest[5]
    const relativeZ = vertexBody.positions[p + 2] - vertexBody.previous[p + 2]
      - (triangleBody.positions[ai + 2] - triangleBody.previous[ai + 2]) * closest[3]
      - (triangleBody.positions[bi + 2] - triangleBody.previous[bi + 2]) * closest[4]
      - (triangleBody.positions[ci + 2] - triangleBody.previous[ci + 2]) * closest[5]
    const normalVelocity = relativeX * nx + relativeY * ny + relativeZ * nz
    const tangentX = relativeX - nx * normalVelocity
    const tangentY = relativeY - ny * normalVelocity
    const tangentZ = relativeZ - nz * normalVelocity
    const tangentLength = Math.hypot(tangentX, tangentY, tangentZ)
    const triangleMaterial = materialForTriangle(triangleBody, triangleOffset / 3)
    const friction = Math.max(0, (vertexBody.materials[0].kineticFriction + triangleMaterial.kineticFriction) * 0.5)
    const tangentScale = tangentLength < EPSILON ? 0 : Math.min(1, friction * normalCorrection / tangentLength)
    const velocityCorrectionX = (nx * Math.max(0, -normalVelocity) - tangentX * tangentScale) / denominator
    const velocityCorrectionY = (ny * Math.max(0, -normalVelocity) - tangentY * tangentScale) / denominator
    const velocityCorrectionZ = (nz * Math.max(0, -normalVelocity) - tangentZ * tangentScale) / denominator
    vertexBody.previous[p] -= velocityCorrectionX * wp
    vertexBody.previous[p + 1] -= velocityCorrectionY * wp
    vertexBody.previous[p + 2] -= velocityCorrectionZ * wp
    for (const [triangleParticle, barycentric, weight] of triangleParticles) {
      const offset = triangleParticle * 3
      const scale = barycentric * weight
      triangleBody.previous[offset] += velocityCorrectionX * scale
      triangleBody.previous[offset + 1] += velocityCorrectionY * scale
      triangleBody.previous[offset + 2] += velocityCorrectionZ * scale
    }
  }
}
