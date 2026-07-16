import { Box3, BufferAttribute, BufferGeometry, Matrix4, Mesh, MeshBasicMaterial, Plane, PlaneGeometry, Sphere, Vector3 } from 'three'
import { describe, expect, it } from 'vitest'

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
