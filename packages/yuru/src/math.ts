import type { Quat, QuatLike, Vec3, Vec3Like } from './types.js'

export const EPSILON = 1e-8

export const readVec3 = (value: Vec3Like): Vec3 => Array.isArray(value)
  ? [value[0]!, value[1]!, value[2]!]
  : [(value as { x: number }).x, (value as { y: number }).y, (value as { z: number }).z]

export const readQuat = (value: QuatLike): Quat => Array.isArray(value)
  ? [value[0]!, value[1]!, value[2]!, value[3]!]
  : [
      (value as { x: number }).x,
      (value as { y: number }).y,
      (value as { z: number }).z,
      (value as { w: number }).w,
    ]

export const distance = (positions: Float32Array, a: number, b: number): number => {
  const ai = a * 3
  const bi = b * 3
  return Math.hypot(
    positions[ai] - positions[bi],
    positions[ai + 1] - positions[bi + 1],
    positions[ai + 2] - positions[bi + 2],
  )
}

export const normalize = (x: number, y: number, z: number): Vec3 => {
  const length = Math.hypot(x, y, z)
  if (length < EPSILON)
    return [0, 1, 0]
  return [x / length, y / length, z / length]
}

export const closestPointOnSegment = (
  px: number,
  py: number,
  pz: number,
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
): [number, number, number, number] => {
  const abx = bx - ax
  const aby = by - ay
  const abz = bz - az
  const denominator = abx * abx + aby * aby + abz * abz
  const t = denominator < EPSILON
    ? 0
    : Math.max(0, Math.min(1, ((px - ax) * abx + (py - ay) * aby + (pz - az) * abz) / denominator))
  return [ax + abx * t, ay + aby * t, az + abz * t, t]
}

/** Returns closest point xyz followed by barycentric weights. */
export const closestPointOnTriangle = (
  px: number,
  py: number,
  pz: number,
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  cx: number,
  cy: number,
  cz: number,
): [number, number, number, number, number, number] => {
  const abx = bx - ax
  const aby = by - ay
  const abz = bz - az
  const acx = cx - ax
  const acy = cy - ay
  const acz = cz - az
  const apx = px - ax
  const apy = py - ay
  const apz = pz - az
  const d1 = abx * apx + aby * apy + abz * apz
  const d2 = acx * apx + acy * apy + acz * apz
  if (d1 <= 0 && d2 <= 0)
    return [ax, ay, az, 1, 0, 0]

  const bpx = px - bx
  const bpy = py - by
  const bpz = pz - bz
  const d3 = abx * bpx + aby * bpy + abz * bpz
  const d4 = acx * bpx + acy * bpy + acz * bpz
  if (d3 >= 0 && d4 <= d3)
    return [bx, by, bz, 0, 1, 0]

  const vc = d1 * d4 - d3 * d2
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3)
    return [ax + v * abx, ay + v * aby, az + v * abz, 1 - v, v, 0]
  }

  const cpx = px - cx
  const cpy = py - cy
  const cpz = pz - cz
  const d5 = abx * cpx + aby * cpy + abz * cpz
  const d6 = acx * cpx + acy * cpy + acz * cpz
  if (d6 >= 0 && d5 <= d6)
    return [cx, cy, cz, 0, 0, 1]

  const vb = d5 * d2 - d1 * d6
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6)
    return [ax + w * acx, ay + w * acy, az + w * acz, 1 - w, 0, w]
  }

  const va = d3 * d6 - d5 * d4
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    return [bx + w * (cx - bx), by + w * (cy - by), bz + w * (cz - bz), 0, 1 - w, w]
  }

  const denominator = 1 / (va + vb + vc)
  const v = vb * denominator
  const w = vc * denominator
  const u = 1 - v - w
  return [
    ax * u + bx * v + cx * w,
    ay * u + by * v + cy * w,
    az * u + bz * v + cz * w,
    u,
    v,
    w,
  ]
}

export const inverseRotate = (point: Vec3, quaternion: Quat): Vec3 => {
  const [x, y, z] = point
  const [qx, qy, qz, qw] = quaternion
  const tx = 2 * (qy * z - qz * y)
  const ty = 2 * (qz * x - qx * z)
  const tz = 2 * (qx * y - qy * x)
  return [
    x - qw * tx + (qy * tz - qz * ty),
    y - qw * ty + (qz * tx - qx * tz),
    z - qw * tz + (qx * ty - qy * tx),
  ]
}

export const rotate = (point: Vec3, quaternion: Quat): Vec3 => {
  const [x, y, z] = point
  const [qx, qy, qz, qw] = quaternion
  const tx = 2 * (qy * z - qz * y)
  const ty = 2 * (qz * x - qx * z)
  const tz = 2 * (qx * y - qy * x)
  return [
    x + qw * tx + (qy * tz - qz * ty),
    y + qw * ty + (qz * tx - qx * tz),
    z + qw * tz + (qx * ty - qy * tx),
  ]
}
