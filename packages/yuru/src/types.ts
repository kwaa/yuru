/* eslint-disable sonarjs/redundant-type-aliases -- Domain IDs prevent accidental API mixups in declarations. */
export type BodyId = number
export interface ClothMaterial {
  bendCompliance: number
  damping: number
  density: number
  drag: number
  kineticFriction: number
  /** Smooths velocity differences between connected particles. */
  laplacianDamping: number
  lift: number
  shearCompliance: number
  staticFriction: number
  stretchCompliance: number
  thickness: number
}
export interface ClothMeshData {
  /** Triangle indices. */
  indices: Uint16Array | Uint32Array
  /** Zero pins a particle. Defaults to one for every particle. */
  inverseMasses?: Float32Array
  /** Packed xyz positions in world space. */
  positions: Float32Array
  /** Optional material index for every triangle. */
  triangleMaterialIndices?: Uint16Array
}

export interface ClothMotionConstraints {
  /** Maximum distance each particle may move from its animated target. */
  maximumDistances: Float32Array
  /** Initial packed xyz targets. Defaults to the mesh positions. */
  targets?: Float32Array
}

/**
 * Numeric limits are expressed in meters per second. `automatic` derives a
 * per-substep displacement cap from cloth thickness and rest-particle spacing.
 */
export type ClothSpeedLimit = 'automatic' | 'unlimited' | number
export type ColliderId = number

export type GrabId = number
export type Quat = readonly [x: number, y: number, z: number, w: number]
export type QuatLike = Quat | Readonly<{ w: number, x: number, y: number, z: number }>

export type Vec3 = readonly [x: number, y: number, z: number]

export type Vec3Like = Readonly<{ x: number, y: number, z: number }> | Vec3

export const DEFAULT_CLOTH_MATERIAL: Readonly<ClothMaterial> = Object.freeze({
  bendCompliance: 1e-6,
  damping: 0.015,
  density: 0.2,
  drag: 0.015,
  kineticFriction: 0.35,
  laplacianDamping: 0.2,
  lift: 0.005,
  shearCompliance: 2e-6,
  staticFriction: 0.45,
  stretchCompliance: 1e-7,
  thickness: 0.008,
})

export interface CollisionFilter {
  group: number
  mask: number
}

export const DEFAULT_COLLISION_FILTER: Readonly<CollisionFilter> = Object.freeze({
  group: 1,
  mask: 0xFFFF_FFFF,
})

export interface CapsuleCollider {
  end: Vec3Like
  radius: number
  start: Vec3Like
  type: 'capsule'
}

export interface ClothBodyDescriptor {
  collisionFilter?: Partial<CollisionFilter>
  /** Ordered garment layer. Higher layers stay on the positive/outside side of lower layers. */
  collisionLayer?: number
  /** Optional axis removed from ordered contact normals, useful for gravity-led layered garments. */
  collisionLayerAxis?: Vec3Like
  id?: string
  materials?: readonly Partial<ClothMaterial>[]
  mesh: ClothMeshData
  /** Per-particle animated motion limits, equivalent to painted cloth max distances. */
  motionConstraints?: ClothMotionConstraints | false
  selfCollision?: boolean
  /** Long-range constraints to connected pinned particles. Defaults on when pins exist. */
  tethers?: boolean
}

export interface ColliderDescriptor {
  collisionFilter?: Partial<CollisionFilter>
  friction?: number
  shape: ColliderShape
  velocity?: Vec3
}

export type ColliderShape = CapsuleCollider | PlaneCollider | RoundedBoxCollider | SphereCollider | TriangleMeshCollider

export interface DirectionalForceField {
  acceleration: Vec3Like
  type: 'directional'
}

export type ForceField = DirectionalForceField | NoiseForceField | PointForceField | VortexForceField | WindForceField

export interface GrabDescriptor {
  body: BodyId
  compliance?: number
  position: Vec3Like
  radius: number
}

export interface NoiseForceField {
  amplitude: number
  frequency: number
  seed?: number
  type: 'noise'
}

export interface PlaneCollider {
  /** Three.Plane-compatible constant. Signed distance is dot(normal, point) + constant. */
  constant?: number
  normal: Vec3Like
  /** Positive distance along the normal. Ignored when constant is supplied. */
  offset?: number
  type: 'plane'
}

export interface PointForceField {
  falloff?: number
  position: Vec3Like
  radius: number
  strength: number
  type: 'point'
}

export type QualityPreset = 'high' | 'low' | 'medium'

export interface QualityProfile {
  collisionEverySubsteps: number
  collisionIterations: number
  continuousCollision: boolean
  maxCatchUpSteps: number
  maxCollisionCandidates: number
  substeps: number
}

export interface RoundedBoxCollider {
  center: Vec3Like
  halfExtents: Vec3Like
  radius: number
  rotation?: QuatLike
  type: 'roundedBox'
}

export interface SphereCollider {
  center: Vec3Like
  radius: number
  type: 'sphere'
}

export interface TriangleMeshCollider {
  indices: Uint16Array | Uint32Array
  positions: Float32Array
  type: 'triangleMesh'
}

export interface VortexForceField {
  axis: Vec3Like
  origin: Vec3Like
  radius: number
  strength: number
  type: 'vortex'
}

export interface WindForceField {
  type: 'wind'
  velocity: Vec3Like
}

export const QUALITY_PRESETS: Readonly<Record<QualityPreset, Readonly<QualityProfile>>> = Object.freeze({
  high: Object.freeze({
    collisionEverySubsteps: 3,
    collisionIterations: 1,
    continuousCollision: true,
    maxCatchUpSteps: 4,
    maxCollisionCandidates: 32,
    substeps: 6,
  }),
  low: Object.freeze({
    collisionEverySubsteps: 2,
    collisionIterations: 1,
    continuousCollision: false,
    maxCatchUpSteps: 2,
    maxCollisionCandidates: 12,
    substeps: 2,
  }),
  medium: Object.freeze({
    collisionEverySubsteps: 4,
    collisionIterations: 1,
    continuousCollision: false,
    maxCatchUpSteps: 3,
    maxCollisionCandidates: 24,
    substeps: 4,
  }),
})

export interface BackendCapabilities {
  continuousCollision: boolean
  kind: BackendKind
  simd: boolean
  threads: boolean
  worker: boolean
}

export type BackendKind = 'cpu' | 'tsl' | 'wasm-single'

export interface BackendStepOptions {
  forceFields: readonly ForceField[]
  gravity: Vec3
  quality: QualityProfile
  speedLimit: ClothSpeedLimit
}

export interface ClothBackend {
  addBody: (descriptor: ClothBodyDescriptor) => BodyId
  addCollider: (descriptor: ColliderDescriptor) => ColliderId
  addGrab: (descriptor: GrabDescriptor) => GrabId
  readonly capabilities: BackendCapabilities
  dispose: () => void
  getPositions: (id: BodyId) => Float32Array
  removeBody: (id: BodyId) => void
  removeCollider: (id: ColliderId) => void
  removeGrab: (id: GrabId) => void
  resetBody: (id: BodyId, positions?: Float32Array) => void
  setMotionConstraintTargets: (id: BodyId, positions: Float32Array) => void
  setParticleTargets: (id: BodyId, indices: Uint32Array, positions: Float32Array) => void
  step: (delta: number, options: BackendStepOptions) => Promise<void> | void
  updateCollider: (id: ColliderId, descriptor: ColliderDescriptor) => void
  updateGrab: (id: GrabId, position: Vec3Like) => void
}

export interface ClothWorldOptions {
  backend?: ClothBackend
  fixedDelta?: number
  forceFields?: readonly ForceField[]
  gravity?: Vec3Like
  quality?: Partial<QualityProfile> | QualityPreset
  speedLimit?: ClothSpeedLimit
}

/** Optional numeric kernel used by CPU backends. */
export interface CpuIntegrationKernel {
  readonly capabilities: BackendCapabilities
  dispose?: () => void
  integrate: (
    positions: Float32Array,
    previous: Float32Array,
    inverseMasses: Float32Array,
    accelerations: Float32Array,
    delta: number,
    damping: number,
    body?: CpuSolverBodyState,
    wind?: Vec3,
  ) => Promise<void> | void
  /** When true, `integrate` applies the supplied wind before structural constraints. */
  readonly integratesAerodynamics?: boolean
  /**
   * When true, `integrate` also performs speed limiting, tethers, distance
   * constraints, and area constraints in their normal solver order.
   */
  readonly integratesStructuralConstraints?: boolean
  /** Releases any persistent native state associated with a removed body. */
  removeBody?: (id: BodyId) => void
  /** Optional vertex/triangle collision batch. Static body data may be cached. */
  solveBodyCollisions?: (
    bodies: readonly CpuSolverBodyState[],
    maxCandidates: number,
  ) => Promise<void> | void
  /** Optional edge/edge collision batch. Static body data may be cached. */
  solveEdgeCollisions?: (
    bodies: readonly CpuSolverBodyState[],
    maxCandidates: number,
  ) => Promise<void> | void
}

/**
 * Packed solver state exposed only to advanced numeric kernels. Static
 * topology arrays may be retained by a kernel so they cross the JS/WASM
 * boundary once when a body is first seen instead of once per substep.
 */
export interface CpuSolverBodyState {
  readonly collisionCellSize: number
  readonly collisionLayer: number
  readonly collisionLayerAxis?: Vec3
  readonly filter: CollisionFilter
  readonly id: BodyId
  readonly indices: Uint16Array | Uint32Array
  readonly initial: Float32Array
  readonly inverseMasses: Float32Array
  readonly materials: readonly ClothMaterial[]
  maximumSelfCollisionDepenetration: number
  maximumSelfCollisionDisplacement: number
  readonly positions: Float32Array
  readonly previous: Float32Array
  readonly selfCollision: boolean
  readonly tethers?: {
    readonly topology: {
      readonly anchors: Uint32Array
      readonly lengths: Float32Array
      readonly particles: Uint32Array
    }
  }
  readonly topology: {
    readonly adjacency: Uint32Array
    readonly adjacencyOffsets: Uint32Array
    readonly edgeKinds: Uint8Array
    readonly edgeParticles: Uint32Array
    readonly edgeRestLengths: Float32Array
    readonly triangleParticles: Uint32Array
    readonly triangleRestAreas: Float32Array
  }
  readonly triangleMaterialIndices?: Uint16Array
}

export interface RuntimeDiagnostics {
  averageStepMs: number
  backend: BackendCapabilities
  droppedTime: number
  fallbackReasons: string[]
  lastStepMs: number
  quality: QualityProfile
  simulatedSteps: number
  speedLimit: ClothSpeedLimit
}
