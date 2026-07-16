import type { ColliderDescriptor, ColliderId, Quat, Vec3 } from 'yuru'

import type { ThreeYuruWorld } from './index.js'

import { Quaternion, Vector3 } from 'three'

export interface RapierColliderLike {
  halfExtents?: () => RapierVector
  halfHeight?: () => number
  indices?: () => Uint32Array
  parent?: () => null | { angvel?: () => RapierVector, linvel: () => RapierVector }
  radius?: () => number
  rotation: () => RapierRotation
  roundRadius?: () => number
  translation: () => RapierVector
  vertices?: () => Float32Array
}

export interface RapierRotation extends RapierVector {
  w: number
}

export interface RapierVector {
  x: number
  y: number
  z: number
}

const asVec3 = (value: RapierVector): Vec3 => [value.x, value.y, value.z]
const asQuat = (value: RapierRotation): Quat => [value.x, value.y, value.z, value.w]

export const colliderFromRapier = (collider: RapierColliderLike): ColliderDescriptor => {
  const translation = collider.translation()
  const rotation = collider.rotation()
  const velocity = collider.parent?.()?.linvel() ?? { x: 0, y: 0, z: 0 }
  if (collider.halfExtents != null) {
    return {
      shape: {
        center: asVec3(translation),
        halfExtents: asVec3(collider.halfExtents()),
        radius: collider.roundRadius?.() ?? 0,
        rotation: asQuat(rotation),
        type: 'roundedBox',
      },
      velocity: asVec3(velocity),
    }
  }
  if (collider.radius != null && collider.halfHeight != null) {
    const quaternion = new Quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
    const axis = new Vector3(0, collider.halfHeight(), 0).applyQuaternion(quaternion)
    return {
      shape: {
        end: [translation.x + axis.x, translation.y + axis.y, translation.z + axis.z],
        radius: collider.radius(),
        start: [translation.x - axis.x, translation.y - axis.y, translation.z - axis.z],
        type: 'capsule',
      },
      velocity: asVec3(velocity),
    }
  }
  if (collider.radius != null) {
    return {
      shape: { center: asVec3(translation), radius: collider.radius(), type: 'sphere' },
      velocity: asVec3(velocity),
    }
  }
  if (collider.vertices != null && collider.indices != null) {
    const vertices = collider.vertices()
    const positions = new Float32Array(vertices.length)
    const point = new Vector3()
    const quaternion = new Quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
    for (let offset = 0; offset < vertices.length; offset += 3) {
      point.fromArray(vertices, offset).applyQuaternion(quaternion).add(translation)
      point.toArray(positions, offset)
    }
    return {
      shape: { indices: collider.indices(), positions, type: 'triangleMesh' },
      velocity: asVec3(velocity),
    }
  }
  throw new TypeError('Unsupported Rapier collider shape')
}

export class RapierColliderBridge {
  readonly collider: RapierColliderLike
  readonly id: ColliderId
  private readonly world: ThreeYuruWorld

  constructor(
    world: ThreeYuruWorld,
    collider: RapierColliderLike,
  ) {
    this.world = world
    this.collider = collider
    this.id = world.core.addCollider(colliderFromRapier(collider))
  }

  dispose(): void {
    this.world.core.removeCollider(this.id)
  }

  update(): void {
    this.world.core.updateCollider(this.id, colliderFromRapier(this.collider))
  }
}
