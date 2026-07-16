import type { InterleavedBufferAttribute } from 'three'

import { BufferAttribute, BufferGeometry, SkinnedMesh } from 'three'

const CLOTH_BONE_NAME = /skirt|coat.?skirt|dress|robe|cape|cloth|sleeve|hem|apron|ribbon|scarf|mantle|[裾裙袖]/i
const RIGID_CLOTHING_NAME = /shoe|boot|heel|sole|glove|button|buckle/i
const WELD_PRECISION = 100_000

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
  result.userData = { ...source.userData }
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

export class ExtractedSkinnedCloth {
  readonly mesh: SkinnedMesh
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
    this.remainderGeometry = subsetGeometry(source.geometry, selected, false)
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

    this.simulationMesh = new SkinnedMesh(simulationGeometry, source.material)
    this.simulationMesh.name = `${source.name || 'SkinnedMesh'}_YuruSimulationProxy`
    this.simulationMesh.bindMode = source.bindMode
    this.simulationMesh.bind(source.skeleton, source.bindMatrix)
    this.simulationMesh.position.copy(source.position)
    this.simulationMesh.quaternion.copy(source.quaternion)
    this.simulationMesh.scale.copy(source.scale)
    this.simulationMesh.visible = false
    this.simulationMesh.frustumCulled = false
    this.simulationMesh.morphTargetInfluences = source.morphTargetInfluences?.slice()
    this.simulationMesh.morphTargetDictionary = source.morphTargetDictionary == null
      ? undefined
      : { ...source.morphTargetDictionary }
    source.parent?.add(this.simulationMesh)
  }

  dispose(): void {
    if (this.disposed)
      return
    this.disposed = true
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
