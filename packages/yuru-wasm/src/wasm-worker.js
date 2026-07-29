import { CPUSolverBackend } from 'yuru'

import { WasmIntegrationKernel } from './index.js'

import * as wasmModule from './generated/single/yuru_wasm_single.js'

const backend = new CPUSolverBackend(new WasmIntegrationKernel(wasmModule))
const bodies = new Map()

const respondWithError = (error, requestId) => {
  const normalized = error instanceof Error ? error : new Error(String(error))
  globalThis.postMessage({
    message: normalized.message,
    requestId,
    stack: normalized.stack,
    type: 'error',
  })
}

const addBody = (data) => {
  const id = backend.addBody(data.descriptor, data.sharedPositions)
  if (id !== data.id)
    throw new Error(`WASM Worker body id desynchronized: ${id} !== ${data.id}`)
  if (data.sharedPositions != null) {
    if (!(data.sharedPositions instanceof Float32Array))
      throw new TypeError('WASM Worker shared positions must be a Float32Array')
    if (data.sharedPositions.length !== backend.getPositions(id).length)
      throw new RangeError(`WASM Worker shared positions do not match body ${id}`)
  }
  bodies.set(id, data.sharedPositions)
}

const step = async (data) => {
  await backend.step(data.delta, data.options)
  if (data.signal == null) {
    const positions = [...bodies.keys()].map(id => [id, backend.getPositions(id).slice()])
    globalThis.postMessage(
      { positions, requestId: data.requestId, type: 'step' },
      positions.map(([, snapshot]) => snapshot.buffer),
    )
    return
  }
  Atomics.store(data.signal, 0, 1)
  Atomics.notify(data.signal, 0)
}

const handleCommand = async (data) => {
  switch (data.type) {
    case 'addBody':
      addBody(data)
      break
    case 'addCollider': {
      const id = backend.addCollider(data.descriptor)
      if (id !== data.id)
        throw new Error(`WASM Worker collider id desynchronized: ${id} !== ${data.id}`)
      break
    }
    case 'addGrab': {
      const id = backend.addGrab(data.descriptor)
      if (id !== data.id)
        throw new Error(`WASM Worker grab id desynchronized: ${id} !== ${data.id}`)
      break
    }
    case 'removeBody':
      backend.removeBody(data.id)
      bodies.delete(data.id)
      break
    case 'removeCollider':
      backend.removeCollider(data.id)
      break
    case 'removeGrab':
      backend.removeGrab(data.id)
      break
    case 'resetBody':
      backend.resetBody(data.id, data.positions)
      break
    case 'setMotionConstraintTargets':
      backend.setMotionConstraintTargets(data.id, data.positions)
      break
    case 'setParticleTargets':
      backend.setParticleTargets(data.id, data.indices, data.positions)
      break
    case 'step':
      await step(data)
      break
    case 'updateCollider':
      backend.updateCollider(data.id, data.descriptor)
      break
    case 'updateGrab':
      backend.updateGrab(data.id, data.position)
      break
    default:
      throw new Error(`Unknown WASM Worker command: ${data.type}`)
  }
}

let commandQueue = Promise.resolve()
globalThis.onmessage = ({ data }) => {
  commandQueue = commandQueue
    .then(() => handleCommand(data))
    .catch(error => respondWithError(error, data.requestId))
}

globalThis.postMessage({ type: 'ready' })
