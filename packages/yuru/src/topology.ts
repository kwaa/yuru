import { distance } from './math.js'

export const BEND_EDGE = 1
export const STRETCH_EDGE = 0

export interface Topology {
  /** CSR adjacency: neighbors for particle i are adjacency[offsets[i]..offsets[i + 1]]. */
  adjacency: Uint32Array
  adjacencyOffsets: Uint32Array
  /** One byte and two packed particle indices per distance constraint. */
  edgeKinds: Uint8Array
  edgeParticles: Uint32Array
  edgeRestLengths: Float32Array
  /** Three packed particle indices and one rest area per triangle. */
  triangleParticles: Uint32Array
  triangleRestAreas: Float32Array
}

const triangleArea = (positions: Float32Array, a: number, b: number, c: number): number => {
  const ai = a * 3
  const bi = b * 3
  const ci = c * 3
  const abx = positions[bi] - positions[ai]
  const aby = positions[bi + 1] - positions[ai + 1]
  const abz = positions[bi + 2] - positions[ai + 2]
  const acx = positions[ci] - positions[ai]
  const acy = positions[ci + 1] - positions[ai + 1]
  const acz = positions[ci + 2] - positions[ai + 2]
  return 0.5 * Math.hypot(
    aby * acz - abz * acy,
    abz * acx - abx * acz,
    abx * acy - aby * acx,
  )
}

export const isAdjacent = (topology: Topology, particle: number, candidate: number): boolean => {
  const start = topology.adjacencyOffsets[particle]
  const end = topology.adjacencyOffsets[particle + 1]
  for (let index = start; index < end; index++) {
    const neighbor = topology.adjacency[index]
    if (neighbor === candidate)
      return true
    if (neighbor > candidate)
      return false
  }
  return false
}

/** Tests the compact two-ring exclusion used by cloth self-collision. */
export const isWithinTwoRings = (topology: Topology, particle: number, candidate: number): boolean => {
  if (particle === candidate || isAdjacent(topology, particle, candidate))
    return true
  const start = topology.adjacencyOffsets[particle]
  const end = topology.adjacencyOffsets[particle + 1]
  for (let index = start; index < end; index++) {
    if (isAdjacent(topology, topology.adjacency[index], candidate))
      return true
  }
  return false
}

export const buildTopology = (positions: Float32Array, indices: Uint16Array | Uint32Array): Topology => {
  const particleCount = positions.length / 3
  const adjacencySets = Array.from({ length: particleCount }, () => new Set<number>())
  const edgeOpposites = new Map<string, { a: number, b: number, opposites: number[] }>()
  const triangleParticles = new Uint32Array(indices.length)
  const triangleRestAreas = new Float32Array(indices.length / 3)

  const addEdge = (a: number, b: number, opposite: number) => {
    const lo = Math.min(a, b)
    const hi = Math.max(a, b)
    const key = `${lo}:${hi}`
    const edge = edgeOpposites.get(key)
    if (edge == null)
      edgeOpposites.set(key, { a: lo, b: hi, opposites: [opposite] })
    else
      edge.opposites.push(opposite)
    adjacencySets[a].add(b)
    adjacencySets[b].add(a)
  }

  for (let offset = 0; offset < indices.length; offset += 3) {
    const a = indices[offset]
    const b = indices[offset + 1]
    const c = indices[offset + 2]
    if (a >= particleCount || b >= particleCount || c >= particleCount)
      throw new RangeError(`Triangle ${offset / 3} references a missing particle`)
    triangleParticles[offset] = a
    triangleParticles[offset + 1] = b
    triangleParticles[offset + 2] = c
    triangleRestAreas[offset / 3] = triangleArea(positions, a, b, c)
    addEdge(a, b, c)
    addEdge(b, c, a)
    addEdge(c, a, b)
  }

  let edgeCount = edgeOpposites.size
  for (const edge of edgeOpposites.values()) {
    if (edge.opposites.length === 2)
      edgeCount++
  }
  const edgeKinds = new Uint8Array(edgeCount)
  const edgeParticles = new Uint32Array(edgeCount * 2)
  const edgeRestLengths = new Float32Array(edgeCount)
  let edgeIndex = 0
  for (const edge of edgeOpposites.values()) {
    edgeParticles[edgeIndex * 2] = edge.a
    edgeParticles[edgeIndex * 2 + 1] = edge.b
    edgeRestLengths[edgeIndex] = distance(positions, edge.a, edge.b)
    edgeIndex++
    if (edge.opposites.length === 2) {
      const [a, b] = edge.opposites
      edgeKinds[edgeIndex] = BEND_EDGE
      edgeParticles[edgeIndex * 2] = a
      edgeParticles[edgeIndex * 2 + 1] = b
      edgeRestLengths[edgeIndex] = distance(positions, a, b)
      edgeIndex++
    }
  }

  const adjacencyOffsets = new Uint32Array(particleCount + 1)
  let adjacencyCount = 0
  for (let particle = 0; particle < particleCount; particle++) {
    adjacencyOffsets[particle] = adjacencyCount
    adjacencyCount += adjacencySets[particle].size
  }
  adjacencyOffsets[particleCount] = adjacencyCount
  const adjacency = new Uint32Array(adjacencyCount)
  let adjacencyIndex = 0
  for (const neighbors of adjacencySets) {
    for (const neighbor of [...neighbors].sort((a, b) => a - b))
      adjacency[adjacencyIndex++] = neighbor
  }

  return {
    adjacency,
    adjacencyOffsets,
    edgeKinds,
    edgeParticles,
    edgeRestLengths,
    triangleParticles,
    triangleRestAreas,
  }
}
