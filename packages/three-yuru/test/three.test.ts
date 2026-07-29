import { Box3, BufferAttribute, BufferGeometry, DynamicDrawUsage, Matrix4, Mesh, MeshBasicMaterial, Plane, PlaneGeometry, Sphere, Vector3 } from 'three'
import { describe, expect, it, vi } from 'vitest'

import { createThreeYuruWorld } from '../src/index.js'
import { colliderFromThree } from '../src/shapes.js'

describe('three mesh binding', () => {
  it('limits render-time catch-up to one fixed simulation step', async () => {
    const world = createThreeYuruWorld({ quality: 'high' })

    await world.update(1)

    expect(world.diagnostics.simulatedSteps).toBe(1)
    expect(world.diagnostics.droppedTime).toBe(0)
    world.dispose()
  })

  it('moves pinned particles with an ordinary Mesh transform', async () => {
    const mesh = new Mesh(new PlaneGeometry(1, 1, 2, 2), new MeshBasicMaterial())
    const world = createThreeYuruWorld({ gravity: [0, -9.81, 0] })
    const controller = world.attachCloth(mesh, { pin: 'top', pinTopRatio: 0.01 })

    mesh.position.y = 1
    await world.update(1 / 60)
    const positions = world.core.getPositions(controller.body)
    for (const index of controller.pinnedIndices)
      expect(positions[index * 3 + 1]).toBeCloseTo(1.5, 5)

    world.dispose()
    mesh.geometry.dispose()
    mesh.material.dispose()
  })

  it('pins the top of every disconnected cloth component', () => {
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new BufferAttribute(new Float32Array([
      0,
      2,
      0,
      1,
      2,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      -1,
      0,
    ]), 3))
    geometry.setIndex([0, 1, 2, 3, 4, 5])
    const material = new MeshBasicMaterial()
    const mesh = new Mesh(geometry, material)
    const world = createThreeYuruWorld()

    const controller = world.attachCloth(mesh, { pinTopRatio: 0.01 })

    expect([...controller.pinnedIndices]).toEqual([0, 1, 3, 4])
    world.dispose()
    geometry.dispose()
    material.dispose()
  })

  it('binds a denser display mesh to a lower resolution simulation proxy', async () => {
    const material = new MeshBasicMaterial()
    const display = new Mesh(new PlaneGeometry(1, 1, 4, 4), material)
    const proxy = new Mesh(new PlaneGeometry(1, 1, 1, 1), material)
    const initialY = display.geometry.getAttribute('position').getY(12)
    const world = createThreeYuruWorld({ gravity: [0, -9.81, 0] })
    const controller = world.attachCloth(display, { pin: false, simulationMesh: proxy })

    await world.update(1 / 60)
    expect(world.core.getPositions(controller.body)).toHaveLength(12)
    expect(display.geometry.getAttribute('position').getY(12)).toBeLessThan(initialY)

    world.dispose()
    display.geometry.dispose()
    proxy.geometry.dispose()
    material.dispose()
  })

  it('uses an explicit proxy map for coincident visual vertices', async () => {
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new BufferAttribute(new Float32Array([
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1,
      0,
      -1,
      0,
      0,
      0,
      -1,
      0,
    ]), 3))
    geometry.setIndex([0, 2, 3, 1, 4, 5])
    const proxyGeometry = geometry.clone()
    proxyGeometry.userData.yuruVisualVertexMap = Uint32Array.from([0, 1, 2, 3, 4, 5])
    const material = new MeshBasicMaterial()
    const display = new Mesh(geometry, material)
    const proxy = new Mesh(proxyGeometry, material)
    const world = createThreeYuruWorld({ gravity: [0, -9.81, 0] })
    world.attachCloth(display, {
      inverseMasses: Float32Array.from([0, 1, 0, 0, 1, 1]),
      pin: false,
      simulationMesh: proxy,
    })

    await world.update(1 / 60)

    expect(display.geometry.getAttribute('position').getY(0)).toBe(0)
    expect(display.geometry.getAttribute('position').getY(1)).toBeLessThan(0)
    world.dispose()
    geometry.dispose()
    proxyGeometry.dispose()
    material.dispose()
  })

  it('updates packed dynamic attributes with Three-compatible vertex normals', () => {
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new BufferAttribute(new Float32Array([
      -1,
      -1,
      0,
      1,
      -1,
      0,
      1,
      1,
      0,
      -1,
      1,
      0,
    ]), 3))
    geometry.setIndex([0, 1, 2, 0, 2, 3])
    const material = new MeshBasicMaterial()
    const mesh = new Mesh(geometry, material)
    mesh.position.set(2, -1, 3)
    mesh.rotation.set(0.2, -0.4, 0.1)
    mesh.scale.set(1.2, 0.8, 1.1)
    const world = createThreeYuruWorld()
    const controller = world.attachCloth(mesh, { pin: false })
    const worldPositions = world.core.getPositions(controller.body).slice()
    worldPositions[1] += 0.5
    worldPositions[5] -= 0.25
    worldPositions[8] += 0.75
    world.core.resetBody(controller.body, worldPositions)

    mesh.updateWorldMatrix(true, false)
    const worldToLocal = new Matrix4().copy(mesh.matrixWorld).invert()
    const expectedGeometry = geometry.clone()
    const expectedPosition = expectedGeometry.getAttribute('position')
    const point = new Vector3()
    for (let index = 0; index < expectedPosition.count; index++) {
      point.fromArray(worldPositions, index * 3).applyMatrix4(worldToLocal)
      expectedPosition.setXYZ(index, point.x, point.y, point.z)
    }
    expectedGeometry.computeVertexNormals()

    controller.syncVisual()

    const position = geometry.getAttribute('position')
    const normal = geometry.getAttribute('normal')
    expect([...position.array]).toEqual([...expectedPosition.array])
    expect([...normal.array]).toEqual([...expectedGeometry.getAttribute('normal').array])
    if (!(position instanceof BufferAttribute) || !(normal instanceof BufferAttribute))
      throw new TypeError('Expected packed buffer attributes')
    expect(position.usage).toBe(DynamicDrawUsage)
    expect(normal.usage).toBe(DynamicDrawUsage)
    expect(position.updateRanges).toEqual([{ count: position.array.length, start: 0 }])
    expect(normal.updateRanges).toEqual([{ count: normal.array.length, start: 0 }])

    controller.syncVisual()
    expect(geometry.getAttribute('normal')).toBe(normal)
    expect(position.updateRanges).toHaveLength(1)
    expect(normal.updateRanges).toHaveLength(1)

    world.dispose()
    expectedGeometry.dispose()
    geometry.dispose()
    material.dispose()
  })

  it('updates a shared simulation world matrix only once for animated targets', () => {
    const geometry = new PlaneGeometry(1, 1, 2, 2)
    const material = new MeshBasicMaterial()
    const mesh = new Mesh(geometry, material)
    const world = createThreeYuruWorld()
    const controller = world.attachCloth(mesh, {
      motionConstraints: { maximumDistances: new Float32Array(9).fill(1) },
      pinTopRatio: 0.01,
    })
    const updateWorldMatrix = vi.spyOn(mesh, 'updateWorldMatrix')

    controller.updateKinematicTargets()

    expect(updateWorldMatrix).toHaveBeenCalledTimes(1)
    world.dispose()
    geometry.dispose()
    material.dispose()
  })
})

describe('three collider conversion', () => {
  it('applies matrix scale to spheres and boxes', () => {
    const matrix = new Matrix4().makeScale(2, 3, 4).setPosition(2, 3, 4)
    const sphere = colliderFromThree(new Sphere(new Vector3(1, 0, 0), 0.5), { matrix })
    const box = colliderFromThree(new Box3(new Vector3(-1, -1, -1), new Vector3(1, 1, 1)), { matrix })

    expect(sphere.shape).toMatchObject({ radius: 2, type: 'sphere' })
    expect(box.shape).toMatchObject({ type: 'roundedBox' })
    if (box.shape.type === 'roundedBox')
      expect(box.shape.halfExtents).toMatchObject({ x: 2, y: 3, z: 4 })
  })

  it('preserves Three.Plane distance convention', () => {
    const collider = colliderFromThree(new Plane(new Vector3(0, 1, 0), -2))
    expect(collider.shape).toMatchObject({ constant: -2, type: 'plane' })
  })
})
