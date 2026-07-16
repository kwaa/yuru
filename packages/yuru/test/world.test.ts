import { afterEach, describe, expect, it, vi } from 'vitest'

import { CPUBackend, createClothWorld } from '../src/index'

const triangle = (inverseMasses = new Float32Array([0, 1, 1])) => ({
  indices: new Uint16Array([0, 1, 2]),
  inverseMasses,
  positions: new Float32Array([0, 1, 0, 1, 1, 0, 0, 0, 0]),
})

describe('clothWorld', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('starts CPUBackend in a dedicated module Worker when the runtime supports it', () => {
    const workers: Array<{ messages: unknown[], options: WorkerOptions, terminated: boolean }> = []
    class MockWorker {
      readonly messages: unknown[] = []
      readonly options: WorkerOptions
      terminated = false

      constructor(_url: URL, options: WorkerOptions) {
        this.options = options
        workers.push(this)
      }

      addEventListener(): void {}

      postMessage(message: unknown): void {
        this.messages.push(message)
      }

      terminate(): void {
        this.terminated = true
      }
    }
    vi.stubGlobal('Worker', MockWorker)

    const backend = new CPUBackend()
    backend.addBody({ mesh: triangle(), selfCollision: false })

    expect(backend.capabilities.worker).toBe(true)
    expect(workers).toHaveLength(1)
    expect(workers[0].options).toMatchObject({ name: 'yuru-cpu', type: 'module' })
    expect(workers[0].messages).toHaveLength(1)
    expect(workers[0].messages[0]).toMatchObject({ type: 'addBody' })
    backend.dispose()
    expect(workers[0].terminated).toBe(true)
  })

  it('uses the CPU backend unless a ClothBackend is injected', () => {
    const defaultWorld = createClothWorld()
    expect(defaultWorld.diagnostics.backend.kind).toBe('cpu')
    defaultWorld.dispose()

    const backend = new CPUBackend({ worker: false })
    const customWorld = createClothWorld({ backend })
    expect(customWorld.backend).toBe(backend)
    customWorld.dispose()
  })

  it('keeps pinned particles fixed while free particles respond to gravity', async () => {
    const world = createClothWorld({ fixedDelta: 1 / 60, quality: 'medium' })
    const body = world.addBody({ mesh: triangle(), selfCollision: false })
    const initial = world.getPositions(body).slice()
    for (let i = 0; i < 30; i++)
      await world.step(1 / 60)
    const positions = world.getPositions(body)
    expect([...positions.subarray(0, 3)]).toEqual([...initial.subarray(0, 3)])
    expect(positions[4]).toBeLessThan(initial[4])
    expect(positions.every(Number.isFinite)).toBe(true)
    world.dispose()
  })

  it('limits self-collision transport speed before collision detection', async () => {
    const speedLimit = 0.6
    const world = createClothWorld({
      fixedDelta: 1 / 60,
      gravity: [0, -600, 0],
      quality: { collisionEverySubsteps: 1, collisionIterations: 0, substeps: 1 },
      speedLimit,
    })
    const body = world.addBody({ mesh: triangle(new Float32Array([1, 1, 1])) })
    const initial = world.getPositions(body).slice()

    await world.step(1 / 60)

    const positions = world.getPositions(body)
    for (let offset = 0; offset < positions.length; offset += 3) {
      expect(Math.hypot(
        positions[offset] - initial[offset],
        positions[offset + 1] - initial[offset + 1],
        positions[offset + 2] - initial[offset + 2],
      )).toBeLessThanOrEqual(speedLimit / 60 + 1e-6)
    }
    expect(world.diagnostics.speedLimit).toBeCloseTo(speedLimit)
    world.dispose()
  })

  it('allows explicitly unlimited cloth speed and rejects invalid numeric limits', async () => {
    expect(() => createClothWorld({ speedLimit: 0 })).toThrow(RangeError)
    const world = createClothWorld({
      fixedDelta: 1 / 60,
      gravity: [0, -600, 0],
      quality: { collisionEverySubsteps: 1, collisionIterations: 0, substeps: 1 },
      speedLimit: 'unlimited',
    })
    const body = world.addBody({ mesh: triangle(new Float32Array([1, 1, 1])) })
    const initialY = world.getPositions(body)[1]

    await world.step(1 / 60)

    expect(initialY - world.getPositions(body)[1]).toBeGreaterThan(0.1)
    expect(() => world.setSpeedLimit(Number.NaN)).toThrow(RangeError)
    world.dispose()
  })

  it('uses geodesic tethers to cap distance from connected pinned particles', async () => {
    const world = createClothWorld({
      fixedDelta: 1 / 60,
      gravity: [0, -14_400, 0],
      quality: { collisionIterations: 0, substeps: 1 },
      speedLimit: 'unlimited',
    })
    const body = world.addBody({
      materials: [{ bendCompliance: 1e6, shearCompliance: 1e6, stretchCompliance: 1e6 }],
      mesh: {
        indices: new Uint16Array([0, 2, 1, 1, 2, 3]),
        inverseMasses: new Float32Array([0, 0, 1, 1]),
        positions: new Float32Array([
          -0.5,
          1,
          0,
          0.5,
          1,
          0,
          -0.5,
          0,
          0,
          0.5,
          0,
          0,
        ]),
      },
      selfCollision: false,
      tethers: true,
    })

    await world.step(1 / 60)

    const positions = world.getPositions(body)
    expect(Math.hypot(positions[6] - positions[0], positions[7] - positions[1], positions[8] - positions[2])).toBeLessThanOrEqual(1.000_001)
    expect(Math.hypot(positions[9] - positions[3], positions[10] - positions[4], positions[11] - positions[5])).toBeLessThanOrEqual(1.000_001)
    world.dispose()
  })

  it('limits particles around animated motion constraint targets', async () => {
    const world = createClothWorld({
      fixedDelta: 1 / 60,
      gravity: [0, -600, 0],
      quality: { collisionIterations: 0, substeps: 1 },
      speedLimit: 'unlimited',
    })
    const maximumDistances = new Float32Array([0.1, 0.1, 0.1])
    const body = world.addBody({
      materials: [{ bendCompliance: 1e6, shearCompliance: 1e6, stretchCompliance: 1e6 }],
      mesh: triangle(new Float32Array([1, 1, 1])),
      motionConstraints: { maximumDistances },
      selfCollision: false,
    })

    await world.step(1 / 60)
    const initialTargets = triangle().positions
    let positions = world.getPositions(body)
    for (let particle = 0; particle < maximumDistances.length; particle++) {
      const offset = particle * 3
      expect(Math.hypot(
        positions[offset] - initialTargets[offset],
        positions[offset + 1] - initialTargets[offset + 1],
        positions[offset + 2] - initialTargets[offset + 2],
      )).toBeLessThanOrEqual(maximumDistances[particle] + 1e-6)
    }

    const movedTargets = initialTargets.slice()
    for (let offset = 1; offset < movedTargets.length; offset += 3)
      movedTargets[offset]++
    world.setMotionConstraintTargets(body, movedTargets)
    await world.step(1 / 60)
    positions = world.getPositions(body)
    for (let particle = 0; particle < maximumDistances.length; particle++) {
      const offset = particle * 3
      expect(Math.hypot(
        positions[offset] - movedTargets[offset],
        positions[offset + 1] - movedTargets[offset + 1],
        positions[offset + 2] - movedTargets[offset + 2],
      )).toBeLessThanOrEqual(maximumDistances[particle] + 1e-6)
    }
    world.dispose()
  })

  it('sweeps fast particles against colliders instead of sampling their midpoint', async () => {
    const createWorld = (continuousCollision: boolean) => createClothWorld({
      fixedDelta: 1 / 60,
      gravity: [14_400, 0, 0],
      quality: { collisionEverySubsteps: 1, collisionIterations: 1, continuousCollision, substeps: 1 },
      speedLimit: 'unlimited',
    })
    const simulate = async (continuousCollision: boolean): Promise<Float32Array> => {
      const world = createWorld(continuousCollision)
      const body = world.addBody({
        materials: [{ thickness: 0.001 }],
        mesh: {
          indices: new Uint16Array([0, 1, 2]),
          positions: new Float32Array([-2, -0.1, 0, -2, 0, 0, -2, 0.1, 0]),
        },
        selfCollision: false,
      })
      world.addCollider({ shape: { center: [0, 0, 0], radius: 0.5, type: 'sphere' } })
      await world.step(1 / 60)
      const result = world.getPositions(body).slice()
      world.dispose()
      return result
    }

    const discrete = await simulate(false)
    const continuous = await simulate(true)
    expect(discrete[3]).toBeGreaterThan(1)
    expect(continuous[3]).toBeLessThan(-0.45)
  })

  it('carries contacted particles with a fast moving capsule', async () => {
    const world = createClothWorld({
      fixedDelta: 1 / 60,
      gravity: [0, 0, 0],
      quality: { collisionEverySubsteps: 1, collisionIterations: 1, continuousCollision: true, substeps: 1 },
      speedLimit: 'unlimited',
    })
    const body = world.addBody({
      materials: [{ thickness: 0.001 }],
      mesh: {
        indices: new Uint16Array([0, 1, 2]),
        positions: new Float32Array([0, -0.1, 0, 0, 0, 0, 0, 0.1, 0]),
      },
      selfCollision: false,
    })
    const collider = world.addCollider({
      shape: { end: [-2, 0, 1], radius: 0.5, start: [-2, 0, -1], type: 'capsule' },
    })
    world.updateCollider(collider, {
      shape: { end: [2, 0, 1], radius: 0.5, start: [2, 0, -1], type: 'capsule' },
    })

    await world.step(1 / 60)

    expect(world.getPositions(body)[3]).toBeGreaterThan(2.45)
    world.dispose()
  })

  it('accepts Three-compatible vector shapes without importing Three', async () => {
    const world = createClothWorld({
      gravity: { x: 0, y: 0, z: 0 },
      quality: { collisionEverySubsteps: 1, substeps: 1 },
    })
    const body = world.addBody({ mesh: triangle(new Float32Array([1, 1, 1])), selfCollision: false })
    world.addCollider({ shape: { center: { x: 0, y: 0.5, z: 0 }, radius: 0.75, type: 'sphere' } })
    await world.step(1 / 60)
    const positions = world.getPositions(body)
    const distance = Math.hypot(positions[0], positions[1] - 0.5, positions[2])
    expect(distance).toBeGreaterThanOrEqual(0.75)
    world.dispose()
  })

  it('resolves initial collider overlap without injecting launch velocity', async () => {
    const world = createClothWorld({ gravity: [0, 0, 0], quality: 'low' })
    const body = world.addBody({
      mesh: {
        indices: new Uint16Array([0, 1, 2]),
        positions: new Float32Array(9),
      },
      selfCollision: false,
    })
    world.addCollider({ shape: { center: [0, 0, 0], radius: 0.5, type: 'sphere' } })

    for (let frame = 0; frame < 30; frame++)
      await world.step(1 / 60)

    const positions = world.getPositions(body)
    for (let offset = 0; offset < positions.length; offset += 3) {
      const distance = Math.hypot(positions[offset], positions[offset + 1], positions[offset + 2])
      expect(distance).toBeGreaterThanOrEqual(0.5)
      expect(distance).toBeLessThan(0.55)
    }
    world.dispose()
  })

  it('treats coincident rest-pose seams as cloth topology', async () => {
    const positions = new Float32Array([
      0,
      1,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
    ])
    const world = createClothWorld({ gravity: [0, 0, 0], quality: 'high' })
    const body = world.addBody({
      mesh: {
        indices: new Uint16Array([0, 1, 2, 3, 4, 5]),
        positions,
      },
    })

    await world.step(1 / 60)

    const simulated = world.getPositions(body)
    for (let index = 0; index < simulated.length; index++)
      expect(simulated[index]).toBeCloseTo(positions[index], 6)
    world.dispose()
  })

  it('moves a volume grab with its updated target and releases cleanly', async () => {
    const world = createClothWorld({ gravity: [0, 0, 0], quality: 'medium' })
    const body = world.addBody({ mesh: triangle(new Float32Array([1, 1, 1])), selfCollision: false })
    const grab = world.addGrab({ body, position: [1, 1, 0], radius: 0.1 })
    world.updateGrab(grab, { x: 1.5, y: 1.5, z: 0 })
    await world.step(1 / 60)
    expect(world.getPositions(body)[3]).toBeGreaterThan(1.25)
    world.removeGrab(grab)
    world.dispose()
  })

  it('drops excess accumulated time instead of spiraling', async () => {
    const world = createClothWorld({ fixedDelta: 1 / 60, quality: { maxCatchUpSteps: 1 } })
    world.addBody({ mesh: triangle(), selfCollision: false })
    await world.step(1)
    expect(world.diagnostics.simulatedSteps).toBe(1)
    expect(world.diagnostics.droppedTime).toBeGreaterThan(0.9)
    world.dispose()
  })

  it('awaits an asynchronous numeric kernel before solving constraints', async () => {
    let completed = false
    const backend = new CPUBackend({
      integrationKernel: {
        capabilities: {
          continuousCollision: true,
          kind: 'tsl',
          simd: false,
          threads: false,
          worker: false,
        },
        integrate: async () => {
          await Promise.resolve()
          completed = true
        },
      },
      worker: false,
    })
    const world = createClothWorld({ backend })
    world.addBody({ mesh: triangle(), selfCollision: false })

    await world.step(1 / 60)
    expect(completed).toBe(true)
    world.dispose()
  })
})
