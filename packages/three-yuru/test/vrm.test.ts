import type { VRM } from '@pixiv/three-vrm'

import { readFile } from 'node:fs/promises'

import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'

import { createWasmBackend } from '../../yuru-wasm/src/index.js'
import { createThreeYuruWorld } from '../src/index.js'
import { attachYuru, detectVRMClothCandidates } from '../src/vrm.js'

const sampleUrl = new URL('../../../examples/vrm/src/assets/AvatarSample_B.vrm', import.meta.url)

const loadOfficialSample = async (): Promise<VRM> => {
  const bytes = await readFile(sampleUrl)
  const loader = new GLTFLoader()
  loader.register(parser => new VRMLoaderPlugin(parser))
  const gltf = await loader.parseAsync(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    '',
  )
  const vrm = gltf.userData.vrm as VRM
  VRMUtils.rotateVRM0(vrm)
  VRMUtils.removeUnnecessaryVertices(vrm.scene)
  VRMUtils.combineSkeletons(vrm.scene)
  VRMUtils.combineMorphs(vrm)
  return vrm
}

describe('official VRM sample discovery', () => {
  beforeAll(() => {
    vi.stubGlobal('self', globalThis)
    vi.stubGlobal('ProgressEvent', class {})
    vi.stubGlobal('createImageBitmap', async () => ({ close: () => {}, height: 1, width: 1 }))
  })

  afterAll(() => vi.unstubAllGlobals())

  it('finds the skirt embedded in the merged body mesh', async () => {
    const vrm = await loadOfficialSample()
    const candidate = detectVRMClothCandidates(vrm)[0]

    expect(candidate?.mesh.name).toBe('Body_(merged)baked_1')
    expect(candidate?.confidence).toBeGreaterThan(0.9)
    expect(candidate?.triangles).toHaveLength(525)
    expect(candidate?.reasons.join(' ')).toContain('secondary bones')
  }, 20_000)

  it('extracts only the detected triangles and restores the source on dispose', async () => {
    const vrm = await loadOfficialSample()
    const source = detectVRMClothCandidates(vrm)[0].mesh
    const originalGeometry = source.geometry
    const world = createThreeYuruWorld()
    const controller = attachYuru(vrm, world)

    expect(controller.status).toBe('ready')
    expect(controller.cloth).toHaveLength(1)
    expect(controller.bodyColliders).toHaveLength(6)
    expect(source.geometry).not.toBe(originalGeometry)
    // Render-only UV/material seams are welded into one simulation proxy.
    expect(world.core.getPositions(controller.cloth[0].body).length / 3).toBe(313)

    controller.dispose()
    expect(source.geometry).toBe(originalGeometry)
    world.dispose()
  }, 20_000)

  it('keeps the official sample skirt bounded during sustained WASM simulation', async () => {
    const vrm = await loadOfficialSample()
    const backend = await createWasmBackend()
    const world = createThreeYuruWorld({ backend, quality: 'high' })
    const controller = attachYuru(vrm, world)
    const body = controller.cloth[0].body
    const initial = world.core.getPositions(body).slice()
    const previous = initial.slice()
    const maximumFrameDisplacements: number[] = []
    const rootMeanSquareDisplacements: number[] = []

    for (let frame = 0; frame < 240; frame++) {
      await controller.update(1 / 60)
      const current = world.core.getPositions(body)
      let maximumFrameDisplacement = 0
      let squaredDisplacement = 0
      for (let offset = 0; offset < current.length; offset += 3) {
        const dx = current[offset] - previous[offset]
        const dy = current[offset + 1] - previous[offset + 1]
        const dz = current[offset + 2] - previous[offset + 2]
        const displacement = dx * dx + dy * dy + dz * dz
        squaredDisplacement += displacement
        maximumFrameDisplacement = Math.max(maximumFrameDisplacement, Math.sqrt(displacement))
      }
      maximumFrameDisplacements.push(maximumFrameDisplacement)
      rootMeanSquareDisplacements.push(Math.sqrt(squaredDisplacement / (current.length / 3)))
      previous.set(current)
    }

    const positions = world.core.getPositions(body)
    let maximumDisplacement = 0
    for (let offset = 0; offset < positions.length; offset += 3) {
      maximumDisplacement = Math.max(maximumDisplacement, Math.hypot(
        positions[offset] - initial[offset],
        positions[offset + 1] - initial[offset + 1],
        positions[offset + 2] - initial[offset + 2],
      ))
    }
    expect(positions.every(Number.isFinite)).toBe(true)
    expect(maximumDisplacement).toBeLessThan(0.75)
    const average = (values: readonly number[]): number => values.reduce((sum, value) => sum + value, 0) / values.length
    const settlingMotion = average(rootMeanSquareDisplacements.slice(15, 30))
    const residualMotion = average(rootMeanSquareDisplacements.slice(-20))
    expect(residualMotion).toBeLessThan(settlingMotion)
    // Fixed acceptance limits: 6 mm worst-particle and 0.2 mm RMS motion per
    // rendered frame during the final third of a second.
    expect.soft(Math.max(...maximumFrameDisplacements.slice(-20))).toBeLessThan(0.006)
    expect.soft(residualMotion).toBeLessThan(0.0002)

    controller.dispose()
    world.dispose()
  }, 20_000)
})
