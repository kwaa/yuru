import type { VRM, VRMHumanBoneName as VRMHumanBoneNameType } from '@pixiv/three-vrm'
import type { BufferGeometry, Material, Object3D } from 'three'
import type { GLTF, GLTFLoaderPlugin, GLTFParser } from 'three/examples/jsm/loaders/GLTFLoader.js'
import type { ColliderDescriptor, ColliderId } from 'yuru'

import type { ThreeClothController, ThreeClothOptions, ThreeYuruWorld } from './index.js'

import { VRMHumanBoneName } from '@pixiv/three-vrm'
import { Mesh, SkinnedMesh, Vector3 } from 'three'
import { DEFAULT_CLOTH_MATERIAL } from 'yuru'

import { ExtractedSkinnedCloth, selectSkinnedCloth } from './skinned-cloth.js'

export interface VRMClothCandidate {
  confidence: number
  mesh: Mesh
  reasons: string[]
  /** Triangle ordinals when clothing is embedded in a merged skinned mesh. */
  triangles?: Uint32Array
}

type SupportedMesh = Mesh<BufferGeometry, Material | Material[]>
type SupportedSkinnedMesh = SkinnedMesh<BufferGeometry, Material | Material[]>

const POSITIVE_NAME = /cloth|skirt|dress|coat|jacket|shirt|pants|sleeve|robe|cape|[服裙袖]/i
const NEGATIVE_NAME = /body|face|skin|eye|teeth|tongue|hair|素体|[顔目髪]/i

const isMesh = (object: Object3D): object is SupportedMesh => object instanceof Mesh
const isSkinnedMesh = (object: Object3D): object is SupportedSkinnedMesh => object instanceof SkinnedMesh

export const detectVRMClothCandidates = (vrm: VRM): VRMClothCandidate[] => {
  const candidates: VRMClothCandidate[] = []
  vrm.scene.traverse((object) => {
    if (!isMesh(object) || object.geometry.getAttribute('position') == null)
      return
    if (isSkinnedMesh(object)) {
      const selection = selectSkinnedCloth(object)
      if (selection != null) {
        candidates.push({
          confidence: 0.95,
          mesh: object,
          reasons: [
            `${selection.clothVertexCount} vertices are weighted to clothing secondary bones`,
            `matched ${selection.clothBoneNames.length} clothing bones inside a merged mesh`,
          ],
          triangles: selection.triangles,
        })
        return
      }
    }
    const reasons: string[] = []
    let score = isSkinnedMesh(object) ? 0.35 : 0.1
    const materialNames = (Array.isArray(object.material) ? object.material : [object.material]).map(material => material.name).join(' ')
    const label = `${object.name} ${materialNames}`
    if (POSITIVE_NAME.test(label)) {
      score += 0.45
      reasons.push('name or material resembles clothing')
    }
    if (NEGATIVE_NAME.test(label)) {
      score -= 0.55
      reasons.push('name or material resembles body, face, eyes, or hair')
    }
    const vertexCount = object.geometry.getAttribute('position').count
    if (vertexCount >= 32 && vertexCount <= 12_000) {
      score += 0.1
      reasons.push('mesh size is suitable for cloth analysis')
    }
    if (object.geometry.getIndex() != null)
      score += 0.05
    candidates.push({ confidence: Math.max(0, Math.min(1, score)), mesh: object, reasons })
  })
  return candidates.sort((a, b) => b.confidence - a.confidence)
}

export interface AttachYuruOptions extends ThreeClothOptions {
  bodyColliders?: boolean
  confidenceThreshold?: number
  meshes?: readonly Mesh[]
}

export type YuruVRMStatus = 'needsConfiguration' | 'ready'

interface BoneColliderSpec {
  end: VRMHumanBoneNameType
  radiusScale: number
  start: VRMHumanBoneNameType
}

const BODY_COLLIDERS: readonly BoneColliderSpec[] = [
  { end: VRMHumanBoneName.Chest, radiusScale: 0.12, start: VRMHumanBoneName.Hips },
  { end: VRMHumanBoneName.RightUpperLeg, radiusScale: 0.09, start: VRMHumanBoneName.LeftUpperLeg },
  { end: VRMHumanBoneName.LeftLowerLeg, radiusScale: 0.065, start: VRMHumanBoneName.LeftUpperLeg },
  { end: VRMHumanBoneName.RightLowerLeg, radiusScale: 0.065, start: VRMHumanBoneName.RightUpperLeg },
  { end: VRMHumanBoneName.LeftFoot, radiusScale: 0.05, start: VRMHumanBoneName.LeftLowerLeg },
  { end: VRMHumanBoneName.RightFoot, radiusScale: 0.05, start: VRMHumanBoneName.RightLowerLeg },
]

type PendingVRMBoneCollider = Omit<VRMBoneCollider, 'id'>

interface VRMBoneCollider {
  end: Object3D
  id: ColliderId
  radius: number
  start: Object3D
}

const capsuleDescriptor = (collider: PendingVRMBoneCollider): ColliderDescriptor => ({
  friction: 0.4,
  shape: {
    end: collider.end.getWorldPosition(new Vector3()),
    radius: collider.radius,
    start: collider.start.getWorldPosition(new Vector3()),
    type: 'capsule',
  },
})

const clothThickness = (options: ThreeClothOptions): number => {
  if (options.materials == null || options.materials.length === 0)
    return DEFAULT_CLOTH_MATERIAL.thickness
  return Math.max(...options.materials.map(material => material.thickness ?? DEFAULT_CLOTH_MATERIAL.thickness))
}

const fitColliderRadius = (
  baseRadius: number,
  start: Object3D,
  end: Object3D,
  cloth: readonly ThreeClothController[],
  world: ThreeYuruWorld,
  thickness: number,
): number => {
  const startPoint = start.getWorldPosition(new Vector3())
  const endPoint = end.getWorldPosition(new Vector3())
  const segment = endPoint.clone().sub(startPoint)
  const segmentLengthSquared = segment.lengthSq()
  const particle = new Vector3()
  const closest = new Vector3()
  const relative = new Vector3()
  let minimumDistance = Number.POSITIVE_INFINITY
  for (const controller of cloth) {
    const positions = world.core.getPositions(controller.body)
    for (let offset = 0; offset < positions.length; offset += 3) {
      particle.fromArray(positions, offset)
      const interpolation = segmentLengthSquared === 0
        ? 0
        : Math.max(0, Math.min(1, relative.copy(particle).sub(startPoint).dot(segment) / segmentLengthSquared))
      closest.copy(startPoint).addScaledVector(segment, interpolation)
      minimumDistance = Math.min(minimumDistance, particle.distanceTo(closest))
    }
  }
  if (!Number.isFinite(minimumDistance))
    return baseRadius
  // A rest-pose overlap makes the structural and collision constraints fight
  // forever. Keep a small numerical margin outside the cloth thickness.
  return Math.max(0, Math.min(baseRadius, minimumDistance - thickness * 1.05))
}

const avatarHeight = (vrm: VRM): number => {
  vrm.scene.updateWorldMatrix(true, true)
  const head = vrm.humanoid.getRawBoneNode(VRMHumanBoneName.Head)
  const leftFoot = vrm.humanoid.getRawBoneNode(VRMHumanBoneName.LeftFoot)
  const rightFoot = vrm.humanoid.getRawBoneNode(VRMHumanBoneName.RightFoot)
  const firstFoot = leftFoot ?? rightFoot
  if (head == null || firstFoot == null)
    return 1.6
  const headPosition = head.getWorldPosition(new Vector3())
  const footPosition = firstFoot.getWorldPosition(new Vector3())
  if (leftFoot != null && rightFoot != null)
    footPosition.add(rightFoot.getWorldPosition(new Vector3())).multiplyScalar(0.5)
  return Math.max(0.5, headPosition.distanceTo(footPosition))
}

export class YuruController {
  readonly bodyColliders: readonly ColliderId[]
  readonly candidates: readonly VRMClothCandidate[]
  readonly cloth: readonly ThreeClothController[]
  readonly status: YuruVRMStatus
  readonly vrm: VRM
  readonly world: ThreeYuruWorld
  private readonly extracted: ExtractedSkinnedCloth[] = []
  private readonly removeBeforeStep: () => void
  private readonly vrmBoneColliders: VRMBoneCollider[] = []

  constructor(
    vrm: VRM,
    world: ThreeYuruWorld,
    options: AttachYuruOptions = {},
  ) {
    this.vrm = vrm
    this.world = world
    this.candidates = detectVRMClothCandidates(vrm)
    const clothOptions = { ...options }
    delete clothOptions.bodyColliders
    delete clothOptions.confidenceThreshold
    delete clothOptions.meshes
    clothOptions.pinTopRatio ??= 0.08
    const threshold = options.confidenceThreshold ?? 0.55
    if (options.meshes != null) {
      this.cloth = options.meshes.map(mesh => world.attachCloth(mesh, clothOptions))
    }
    else {
      this.cloth = this.candidates
        .filter(candidate => candidate.confidence >= threshold)
        .map((candidate) => {
          if (candidate.triangles == null || !isSkinnedMesh(candidate.mesh))
            return world.attachCloth(candidate.mesh, clothOptions)
          const extracted = new ExtractedSkinnedCloth(candidate.mesh, candidate.triangles)
          this.extracted.push(extracted)
          return world.attachCloth(extracted.mesh, {
            ...clothOptions,
            simulationMesh: clothOptions.simulationMesh ?? extracted.simulationMesh,
          })
        })
    }
    if (options.bodyColliders !== false) {
      const height = avatarHeight(vrm)
      const thickness = clothThickness(clothOptions)
      for (const spec of BODY_COLLIDERS) {
        const start = vrm.humanoid.getRawBoneNode(spec.start)
        const end = vrm.humanoid.getRawBoneNode(spec.end)
        if (start == null || end == null)
          continue
        const radius = fitColliderRadius(height * spec.radiusScale, start, end, this.cloth, world, thickness)
        if (radius <= 0)
          continue
        const base = { end, radius, start }
        this.vrmBoneColliders.push({ ...base, id: world.core.addCollider(capsuleDescriptor(base)) })
      }
    }
    this.bodyColliders = this.vrmBoneColliders.map(collider => collider.id)
    this.removeBeforeStep = world.addBeforeStep(() => this.updateBodyColliders())
    this.status = this.cloth.length > 0 ? 'ready' : 'needsConfiguration'
  }

  dispose(): void {
    this.removeBeforeStep()
    for (const controller of this.cloth)
      controller.dispose()
    for (const extracted of this.extracted)
      extracted.dispose()
    for (const collider of this.vrmBoneColliders)
      this.world.core.removeCollider(collider.id)
  }

  async update(delta: number): Promise<void> {
    return this.world.update(delta)
  }

  private updateBodyColliders(): void {
    this.vrm.scene.updateWorldMatrix(true, true)
    for (const collider of this.vrmBoneColliders)
      this.world.core.updateCollider(collider.id, capsuleDescriptor(collider))
  }
}

export const attachYuru = (vrm: VRM, world: ThreeYuruWorld, options: AttachYuruOptions = {}): YuruController =>
  new YuruController(vrm, world, options)

export interface YuruVRMLoaderPluginOptions {
  analyze?: boolean
}

/** Adds conservative Yuru candidate diagnostics after the regular VRM plugin. */
export class YuruVRMLoaderPlugin implements GLTFLoaderPlugin {
  readonly name = 'YuruVRMLoaderPlugin'
  readonly options: YuruVRMLoaderPluginOptions
  readonly parser: GLTFParser

  constructor(
    parser: GLTFParser,
    options: YuruVRMLoaderPluginOptions = {},
  ) {
    this.parser = parser
    this.options = options
  }

  afterRoot(gltf: GLTF): null {
    const vrm = gltf.userData.vrm as undefined | VRM
    if (vrm != null && this.options.analyze !== false)
      gltf.userData.yuruClothCandidates = detectVRMClothCandidates(vrm)
    return null
  }
}
