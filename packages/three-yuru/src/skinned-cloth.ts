import type { InterleavedBufferAttribute } from 'three'

import { BufferAttribute, BufferGeometry, SkinnedMesh } from 'three'

const CLOTH_BONE_NAME = /skirt|coat.?skirt|dress|robe|cape|cloth|sleeve|hem|apron|ribbon|scarf|mantle|[裾裙袖]/i
const RIGID_CLOTHING_NAME = /shoe|boot|heel|sole|glove|button|buckle/i
const WELD_PRECISION = 100_000
const COLLISION_NEIGHBORHOOD_MINIMUM = 0.03
export const YURU_VISUAL_VERTEX_MAP = 'yuruVisualVertexMap'

export interface SkinnedClothSelection {
  clothBoneNames: readonly string[]
  clothVertexCount: number
  triangles: Uint32Array
}

const geometryIndex = (geometry: BufferGeometry): ArrayLike<number> => {
  const index = geometry.getIndex()
  if (index != null)
    return index.array
  const count = geometry.getAttribute('position').count
  return Uint32Array.from({ length: count }, (_, item) => item)
}

/** Detects triangles driven by secondary clothing bones inside a merged body mesh. */
// eslint-disable-next-line sonarjs/cognitive-complexity, sonarjs/function-return-type
export const selectSkinnedCloth = (mesh: SkinnedMesh): null | SkinnedClothSelection => {
  const skinIndex = mesh.geometry.getAttribute('skinIndex')
  const skinWeight = mesh.geometry.getAttribute('skinWeight')
  const position = mesh.geometry.getAttribute('position')
  if (skinIndex == null || skinWeight == null || position == null)
    return null

  const clothBones = new Set<number>()
  const clothBoneNames: string[] = []
  for (let index = 0; index < mesh.skeleton.bones.length; index++) {
    const name = mesh.skeleton.bones[index]?.name ?? ''
    if (CLOTH_BONE_NAME.test(name) && !RIGID_CLOTHING_NAME.test(name)) {
      clothBones.add(index)
      clothBoneNames.push(name)
    }
  }
  if (clothBones.size === 0)
    return null

  const flexibility = new Float32Array(position.count)
  let clothVertexCount = 0
  for (let vertex = 0; vertex < position.count; vertex++) {
    let score = 0
    for (let component = 0; component < Math.min(4, skinIndex.itemSize, skinWeight.itemSize); component++) {
      if (clothBones.has(Math.round(skinIndex.getComponent(vertex, component))))
        score += skinWeight.getComponent(vertex, component)
    }
    flexibility[vertex] = score
    if (score >= 0.025)
      clothVertexCount++
  }
  if (clothVertexCount < 4)
    return null

  const index = geometryIndex(mesh.geometry)
  const triangles: number[] = []
  for (let offset = 0; offset + 2 < index.length; offset += 3) {
    const a = index[offset]
    const b = index[offset + 1]
    const c = index[offset + 2]
    const maximum = Math.max(flexibility[a], flexibility[b], flexibility[c])
    const total = flexibility[a] + flexibility[b] + flexibility[c]
    // Keep seam triangles where one vertex is rigidly attached to the body.
    if (maximum >= 0.025 && total >= 0.05)
      triangles.push(offset / 3)
  }
  if (triangles.length < 2)
    return null
  return { clothBoneNames, clothVertexCount, triangles: Uint32Array.from(triangles) }
}

const materialIndexAt = (geometry: BufferGeometry, indexOffset: number): number => {
  for (const group of geometry.groups) {
    if (indexOffset >= group.start && indexOffset < group.start + group.count)
      return group.materialIndex ?? 0
  }
  return 0
}

const appendAttributeKey = (
  parts: number[],
  attribute: BufferAttribute | InterleavedBufferAttribute | undefined,
  vertex: number,
  precision: number,
): void => {
  if (attribute == null)
    return
  for (let component = 0; component < attribute.itemSize; component++)
    parts.push(Math.round(attribute.getComponent(vertex, component) * precision))
}

const simulationVertexKey = (geometry: BufferGeometry, vertex: number): string => {
  const parts: number[] = []
  appendAttributeKey(parts, geometry.getAttribute('position'), vertex, WELD_PRECISION)
  appendAttributeKey(parts, geometry.getAttribute('skinIndex'), vertex, 1)
  appendAttributeKey(parts, geometry.getAttribute('skinWeight'), vertex, WELD_PRECISION)
  for (const attribute of geometry.morphAttributes.position ?? [])
    appendAttributeKey(parts, attribute, vertex, WELD_PRECISION)
  return parts.join(',')
}

const copyGeometryVertices = (
  source: BufferGeometry,
  sourceVertices: readonly number[],
  result: BufferGeometry,
): void => {
  for (const [name, attribute] of Object.entries(source.attributes)) {
    const values = new Float32Array(sourceVertices.length * attribute.itemSize)
    for (let vertex = 0; vertex < sourceVertices.length; vertex++) {
      for (let component = 0; component < attribute.itemSize; component++)
        values[vertex * attribute.itemSize + component] = attribute.getComponent(sourceVertices[vertex], component)
    }
    result.setAttribute(name, new BufferAttribute(values, attribute.itemSize))
  }
  for (const [name, attributes] of Object.entries(source.morphAttributes)) {
    result.morphAttributes[name as 'color' | 'normal' | 'position'] = attributes.map((attribute) => {
      const values = new Float32Array(sourceVertices.length * attribute.itemSize)
      for (let vertex = 0; vertex < sourceVertices.length; vertex++) {
        for (let component = 0; component < attribute.itemSize; component++)
          values[vertex * attribute.itemSize + component] = attribute.getComponent(sourceVertices[vertex], component)
      }
      return new BufferAttribute(values, attribute.itemSize)
    })
  }
  result.morphTargetsRelative = source.morphTargetsRelative
}

/** Builds an unrendered simulation proxy by removing render-only seam splits. */
export const weldSkinnedSimulationGeometry = (source: BufferGeometry): BufferGeometry => {
  const sourcePosition = source.getAttribute('position')
  if (sourcePosition == null)
    throw new Error('A skinned cloth simulation mesh requires positions')
  const sourceIndex = geometryIndex(source)
  const oldToNew = new Uint32Array(sourcePosition.count)
  const keyToVertex = new Map<string, number>()
  const sourceVertices: number[] = []
  for (let vertex = 0; vertex < sourcePosition.count; vertex++) {
    const key = simulationVertexKey(source, vertex)
    let mapped = keyToVertex.get(key)
    if (mapped == null) {
      mapped = sourceVertices.length
      keyToVertex.set(key, mapped)
      sourceVertices.push(vertex)
    }
    oldToNew[vertex] = mapped
  }

  const indices: number[] = []
  const triangles = new Set<string>()
  for (let offset = 0; offset + 2 < sourceIndex.length; offset += 3) {
    const a = oldToNew[sourceIndex[offset]]
    const b = oldToNew[sourceIndex[offset + 1]]
    const c = oldToNew[sourceIndex[offset + 2]]
    if (a === b || b === c || c === a)
      continue
    const key = [a, b, c].sort((first, second) => first - second).join(',')
    if (triangles.has(key))
      continue
    triangles.add(key)
    indices.push(a, b, c)
  }

  const result = new BufferGeometry()
  result.name = `${source.name}_YuruSimulationProxy`
  // Preserve the exact source-to-proxy correspondence. Position-only nearest
  // point binding is ambiguous at UV/material seams and at coincident vertices
  // with different skin weights; once those proxy particles separate it can
  // fold an otherwise valid render triangle onto the wrong side of a garment.
  result.userData = { ...source.userData, [YURU_VISUAL_VERTEX_MAP]: oldToNew }
  copyGeometryVertices(source, sourceVertices, result)
  result.setIndex(new BufferAttribute(
    sourceVertices.length > 0xFFFF ? Uint32Array.from(indices) : Uint16Array.from(indices),
    1,
  ))
  result.computeBoundingBox()
  result.computeBoundingSphere()
  return result
}

const subsetGeometry = (
  source: BufferGeometry,
  selectedTriangles: ReadonlySet<number>,
  includeSelected: boolean,
): BufferGeometry => {
  const sourceIndex = geometryIndex(source)
  const oldToNew = new Map<number, number>()
  const sourceVertices: number[] = []
  const indices: number[] = []
  const materialIndices: number[] = []

  for (let triangle = 0; triangle * 3 + 2 < sourceIndex.length; triangle++) {
    if (selectedTriangles.has(triangle) !== includeSelected)
      continue
    for (let corner = 0; corner < 3; corner++) {
      const sourceVertex = sourceIndex[triangle * 3 + corner]
      let mapped = oldToNew.get(sourceVertex)
      if (mapped == null) {
        mapped = sourceVertices.length
        oldToNew.set(sourceVertex, mapped)
        sourceVertices.push(sourceVertex)
      }
      indices.push(mapped)
    }
    materialIndices.push(materialIndexAt(source, triangle * 3))
  }

  const result = new BufferGeometry()
  result.name = `${source.name}_${includeSelected ? 'YuruCloth' : 'YuruRemainder'}`
  result.userData = { ...source.userData }
  copyGeometryVertices(source, sourceVertices, result)
  result.setIndex(new BufferAttribute(
    sourceVertices.length > 0xFFFF ? Uint32Array.from(indices) : Uint16Array.from(indices),
    1,
  ))

  let runMaterial = materialIndices[0]
  let runStart = 0
  for (let triangle = 1; triangle <= materialIndices.length; triangle++) {
    const material = materialIndices[triangle]
    if (material !== runMaterial) {
      result.addGroup(runStart * 3, (triangle - runStart) * 3, runMaterial ?? 0)
      runStart = triangle
      runMaterial = material
    }
  }
  result.computeBoundingBox()
  result.computeBoundingSphere()
  return result
}

const positionKey = (position: BufferAttribute | InterleavedBufferAttribute, vertex: number): string =>
  `${Math.round(position.getX(vertex) * WELD_PRECISION)},${Math.round(position.getY(vertex) * WELD_PRECISION)},${Math.round(position.getZ(vertex) * WELD_PRECISION)}`

const weldedComponentRoots = (
  position: BufferAttribute | InterleavedBufferAttribute,
  index: ArrayLike<number>,
): Int32Array => {
  const keyToVertex = new Map<string, number>()
  const welded = new Int32Array(position.count)
  for (let vertex = 0; vertex < position.count; vertex++) {
    const key = positionKey(position, vertex)
    let mapped = keyToVertex.get(key)
    if (mapped == null) {
      mapped = keyToVertex.size
      keyToVertex.set(key, mapped)
    }
    welded[vertex] = mapped
  }
  const parents = Int32Array.from({ length: keyToVertex.size }, (_, item) => item)
  const find = (item: number): number => {
    let root = item
    while (parents[root] !== root)
      root = parents[root]
    while (parents[item] !== item) {
      const parent = parents[item]
      parents[item] = root
      item = parent
    }
    return root
  }
  const join = (first: number, second: number): void => {
    const firstRoot = find(first)
    const secondRoot = find(second)
    if (firstRoot !== secondRoot)
      parents[secondRoot] = firstRoot
  }
  for (let offset = 0; offset + 2 < index.length; offset += 3) {
    join(welded[index[offset]], welded[index[offset + 1]])
    join(welded[index[offset + 1]], welded[index[offset + 2]])
  }
  return Int32Array.from(welded, item => find(item))
}

const collisionNeighborhood = (
  source: BufferGeometry,
  selectedTriangles: ReadonlySet<number>,
// eslint-disable-next-line sonarjs/cognitive-complexity
): Set<number> => {
  const position = source.getAttribute('position')
  const index = geometryIndex(source)
  const componentRoots = weldedComponentRoots(position, index)
  const selectedComponents = new Set<number>()
  const selectedPositions = new Set<string>()
  const selectedRadii: number[] = []
  const minimum = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY]
  const maximum = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY]
  for (const triangle of selectedTriangles) {
    for (let corner = 0; corner < 3; corner++) {
      const vertex = index[triangle * 3 + corner]
      selectedComponents.add(componentRoots[vertex])
      selectedPositions.add(positionKey(position, vertex))
      for (let axis = 0; axis < 3; axis++) {
        const value = position.getComponent(vertex, axis)
        minimum[axis] = Math.min(minimum[axis], value)
        maximum[axis] = Math.max(maximum[axis], value)
      }
    }
  }
  const diagonal = Math.hypot(
    maximum[0] - minimum[0],
    maximum[1] - minimum[1],
    maximum[2] - minimum[2],
  )
  const padding = Math.max(COLLISION_NEIGHBORHOOD_MINIMUM, diagonal * 0.1)
  const centerX = (minimum[0] + maximum[0]) * 0.5
  const centerZ = (minimum[2] + maximum[2]) * 0.5
  for (const triangle of selectedTriangles) {
    for (let corner = 0; corner < 3; corner++) {
      const vertex = index[triangle * 3 + corner]
      selectedRadii.push(Math.hypot(position.getX(vertex) - centerX, position.getZ(vertex) - centerZ))
    }
  }
  selectedRadii.sort((a, b) => a - b)
  const layerRadius = selectedRadii[Math.floor(selectedRadii.length * 0.5)] ?? 0
  const result = new Set<number>()
  for (let triangle = 0; triangle * 3 + 2 < index.length; triangle++) {
    if (selectedTriangles.has(triangle))
      continue
    let sharesSelectedPosition = false
    let overlaps = true
    for (let axis = 0; axis < 3; axis++) {
      let triangleMinimum = Number.POSITIVE_INFINITY
      let triangleMaximum = Number.NEGATIVE_INFINITY
      for (let corner = 0; corner < 3; corner++) {
        const vertex = index[triangle * 3 + corner]
        sharesSelectedPosition ||= selectedPositions.has(positionKey(position, vertex))
        const value = position.getComponent(vertex, axis)
        triangleMinimum = Math.min(triangleMinimum, value)
        triangleMaximum = Math.max(triangleMaximum, value)
      }
      overlaps &&= triangleMaximum >= minimum[axis] - padding && triangleMinimum <= maximum[axis] + padding
    }
    // Adjacent seam triangles are the attachment, not a second garment layer.
    if (!overlaps || sharesSelectedPosition)
      continue
    const a = index[triangle * 3]
    const b = index[triangle * 3 + 1]
    const c = index[triangle * 3 + 2]
    const centroidX = (position.getX(a) + position.getX(b) + position.getX(c)) / 3
    const centroidZ = (position.getZ(a) + position.getZ(b) + position.getZ(c)) / 3
    const abx = position.getX(b) - position.getX(a)
    const aby = position.getY(b) - position.getY(a)
    const abz = position.getZ(b) - position.getZ(a)
    const acx = position.getX(c) - position.getX(a)
    const acy = position.getY(c) - position.getY(a)
    const acz = position.getZ(c) - position.getZ(a)
    const normalX = aby * acz - abz * acy
    const normalZ = abx * acy - aby * acx
    // Merged character garments can contain both sides of a thin shell.
    // Retain only the outward-facing surface so ordered contacts do not
    // receive contradictory normals from the coat's inner duplicate.
    if (normalX * (centroidX - centerX) + normalZ * (centroidZ - centerZ) <= 0)
      continue
    const radius = Math.hypot(centroidX - centerX, centroidZ - centerZ)
    const sameGarmentIsland = selectedComponents.has(componentRoots[a])
    if (!sameGarmentIsland && radius >= layerRadius * 0.8)
      result.add(triangle)
  }
  return result
}

const collisionGeometry = (
  source: BufferGeometry,
  triangles: ReadonlySet<number>,
): BufferGeometry => {
  const subset = subsetGeometry(source, triangles, true)
  const result = weldSkinnedSimulationGeometry(subset)
  subset.dispose()
  return result
}

const hiddenSkinnedMesh = (source: SkinnedMesh, geometry: BufferGeometry, suffix: string): SkinnedMesh => {
  const mesh = new SkinnedMesh(geometry, source.material)
  mesh.name = `${source.name || 'SkinnedMesh'}_${suffix}`
  mesh.bindMode = source.bindMode
  mesh.bind(source.skeleton, source.bindMatrix)
  mesh.position.copy(source.position)
  mesh.quaternion.copy(source.quaternion)
  mesh.scale.copy(source.scale)
  mesh.visible = false
  mesh.frustumCulled = false
  mesh.morphTargetInfluences = source.morphTargetInfluences?.slice()
  mesh.morphTargetDictionary = source.morphTargetDictionary == null
    ? undefined
    : { ...source.morphTargetDictionary }
  source.parent?.add(mesh)
  return mesh
}

const topBoundaryPins = (geometry: BufferGeometry): Uint32Array => {
  const position = geometry.getAttribute('position')
  const index = geometryIndex(geometry)
  const edgeCounts = new Map<string, { a: number, b: number, count: number }>()
  const add = (first: number, second: number): void => {
    const a = Math.min(first, second)
    const b = Math.max(first, second)
    const key = `${a}:${b}`
    const edge = edgeCounts.get(key)
    if (edge == null)
      edgeCounts.set(key, { a, b, count: 1 })
    else
      edge.count++
  }
  for (let offset = 0; offset + 2 < index.length; offset += 3) {
    add(index[offset], index[offset + 1])
    add(index[offset + 1], index[offset + 2])
    add(index[offset + 2], index[offset])
  }
  let minimumY = Number.POSITIVE_INFINITY
  let maximumY = Number.NEGATIVE_INFINITY
  for (let vertex = 0; vertex < position.count; vertex++) {
    minimumY = Math.min(minimumY, position.getY(vertex))
    maximumY = Math.max(maximumY, position.getY(vertex))
  }
  const threshold = maximumY - (maximumY - minimumY) * 0.15
  const pins = new Set<number>()
  for (const edge of edgeCounts.values()) {
    if (edge.count !== 1)
      continue
    if (position.getY(edge.a) >= threshold)
      pins.add(edge.a)
    if (position.getY(edge.b) >= threshold)
      pins.add(edge.b)
  }
  return Uint32Array.from([...pins].sort((a, b) => a - b))
}

export class ExtractedSkinnedCloth {
  readonly collisionLayers: readonly { mesh: SkinnedMesh, offset: 1 }[]
  readonly mesh: SkinnedMesh
  readonly pinnedIndices: Uint32Array
  readonly simulationMesh: SkinnedMesh

  private disposed = false
  private readonly originalGeometry: BufferGeometry
  private readonly remainderGeometry: BufferGeometry
  private readonly source: SkinnedMesh

  constructor(
    source: SkinnedMesh,
    triangles: Uint32Array,
  ) {
    this.source = source
    this.originalGeometry = source.geometry
    const selected = new Set(triangles)
    const clothGeometry = subsetGeometry(source.geometry, selected, true)
    const simulationGeometry = weldSkinnedSimulationGeometry(clothGeometry)
    const collisionTriangles = collisionNeighborhood(this.originalGeometry, selected)
    const outerCollisionGeometry = collisionTriangles.size === 0
      ? undefined
      : collisionGeometry(this.originalGeometry, collisionTriangles)
    this.remainderGeometry = subsetGeometry(this.originalGeometry, selected, false)
    source.geometry = this.remainderGeometry
    source.updateMorphTargets()

    this.mesh = new SkinnedMesh(clothGeometry, source.material)
    this.mesh.name = `${source.name || 'SkinnedMesh'}_YuruClothSource`
    this.mesh.bindMode = source.bindMode
    this.mesh.bind(source.skeleton, source.bindMatrix)
    this.mesh.position.copy(source.position)
    this.mesh.quaternion.copy(source.quaternion)
    this.mesh.scale.copy(source.scale)
    this.mesh.renderOrder = source.renderOrder
    this.mesh.castShadow = source.castShadow
    this.mesh.receiveShadow = source.receiveShadow
    this.mesh.frustumCulled = false
    this.mesh.morphTargetInfluences = source.morphTargetInfluences?.slice()
    this.mesh.morphTargetDictionary = source.morphTargetDictionary == null
      ? undefined
      : { ...source.morphTargetDictionary }
    source.parent?.add(this.mesh)

    this.simulationMesh = hiddenSkinnedMesh(source, simulationGeometry, 'YuruSimulationProxy')
    this.pinnedIndices = topBoundaryPins(simulationGeometry)
    this.collisionLayers = [
      outerCollisionGeometry == null ? undefined : { mesh: hiddenSkinnedMesh(source, outerCollisionGeometry, 'YuruOuterCollisionLayer'), offset: 1 as const },
    ].filter((layer): layer is { mesh: SkinnedMesh, offset: 1 } => layer != null)
  }

  dispose(): void {
    if (this.disposed)
      return
    this.disposed = true
    for (const layer of this.collisionLayers) {
      layer.mesh.removeFromParent()
      layer.mesh.geometry.dispose()
    }
    this.simulationMesh.removeFromParent()
    this.simulationMesh.geometry.dispose()
    this.mesh.removeFromParent()
    this.mesh.geometry.dispose()
    if (this.source.geometry === this.remainderGeometry) {
      this.source.geometry = this.originalGeometry
      this.source.updateMorphTargets()
    }
    this.remainderGeometry.dispose()
  }
}
