import type { Matrix4 } from 'three'
import type { ColliderDescriptor, ColliderShape } from 'yuru'

import { Box3, BufferGeometry, Plane, Quaternion, Sphere, Vector3 } from 'three'

export interface CapsuleLike {
  end: Vector3
  radius: number
  start: Vector3
}

export type ThreeColliderShape = Box3 | BufferGeometry | CapsuleLike | Plane | Sphere

export interface ThreeShapeOptions extends Omit<ColliderDescriptor, 'shape'> {
  matrix?: Matrix4
  roundedRadius?: number
}

const transformPoint = (target: Vector3, matrix?: Matrix4): Vector3 => matrix == null ? target : target.applyMatrix4(matrix)

export const geometryToTriangleMeshShape = (geometry: BufferGeometry, matrix?: Matrix4): ColliderShape => {
  const position = geometry.getAttribute('position')
  if (position == null)
    throw new Error('BufferGeometry requires a position attribute')
  const positions = new Float32Array(position.count * 3)
  const point = new Vector3()
  for (let index = 0; index < position.count; index++) {
    point.fromBufferAttribute(position, index)
    transformPoint(point, matrix)
    point.toArray(positions, index * 3)
  }
  const sourceIndices = geometry.getIndex()
  const indices = sourceIndices == null
    ? Uint32Array.from({ length: position.count }, (_, index) => index)
    : Uint32Array.from(sourceIndices.array)
  if (indices.length % 3 !== 0)
    throw new Error('Only triangle BufferGeometry is supported')
  return { indices, positions, type: 'triangleMesh' }
}

export const colliderFromThree = (shape: ThreeColliderShape, options: ThreeShapeOptions = {}): ColliderDescriptor => {
  let descriptor: ColliderShape
  if (shape instanceof Sphere) {
    const center = transformPoint(shape.center.clone(), options.matrix)
    const scale = options.matrix?.getMaxScaleOnAxis() ?? 1
    descriptor = { center, radius: shape.radius * scale, type: 'sphere' }
  }
  else if (shape instanceof Plane) {
    const plane = shape.clone()
    if (options.matrix != null)
      plane.applyMatrix4(options.matrix)
    descriptor = { constant: plane.constant, normal: plane.normal, type: 'plane' }
  }
  else if (shape instanceof Box3) {
    const center = shape.getCenter(new Vector3())
    const halfExtents = shape.getSize(new Vector3()).multiplyScalar(0.5)
    const rotation = new Quaternion()
    if (options.matrix != null) {
      const scale = new Vector3()
      transformPoint(center, options.matrix)
      options.matrix.decompose(new Vector3(), rotation, scale)
      halfExtents.multiply(scale.set(Math.abs(scale.x), Math.abs(scale.y), Math.abs(scale.z)))
    }
    descriptor = {
      center,
      halfExtents,
      radius: options.roundedRadius ?? 0,
      rotation,
      type: 'roundedBox',
    }
  }
  else if (shape instanceof BufferGeometry) {
    descriptor = geometryToTriangleMeshShape(shape, options.matrix)
  }
  else if ('start' in shape && 'end' in shape && 'radius' in shape) {
    descriptor = {
      end: transformPoint(shape.end.clone(), options.matrix),
      radius: shape.radius * (options.matrix?.getMaxScaleOnAxis() ?? 1),
      start: transformPoint(shape.start.clone(), options.matrix),
      type: 'capsule',
    }
  }
  else {
    throw new TypeError('Unsupported Three collider shape')
  }
  const colliderOptions = { ...options }
  delete colliderOptions.matrix
  delete colliderOptions.roundedRadius
  return { ...colliderOptions, shape: descriptor }
}
