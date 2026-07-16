import { CPUSolverBackend } from './cpu-solver-backend.js'

const backend = new CPUSolverBackend()
const bodies = new Set()

const respondWithError = (error, requestId) => {
  const normalized = error instanceof Error ? error : new Error(String(error))
  globalThis.postMessage({
    message: normalized.message,
    requestId,
    stack: normalized.stack,
    type: 'error',
  })
}

globalThis.onmessage = async ({ data }) => {
  try {
    switch (data.type) {
      case 'addBody': {
        const id = backend.addBody(data.descriptor)
        if (id !== data.id)
          throw new Error(`CPU Worker body id desynchronized: ${id} !== ${data.id}`)
        bodies.add(id)
        break
      }
      case 'addCollider': {
        const id = backend.addCollider(data.descriptor)
        if (id !== data.id)
          throw new Error(`CPU Worker collider id desynchronized: ${id} !== ${data.id}`)
        break
      }
      case 'addGrab': {
        const id = backend.addGrab(data.descriptor)
        if (id !== data.id)
          throw new Error(`CPU Worker grab id desynchronized: ${id} !== ${data.id}`)
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
      case 'setParticleTargets':
        backend.setParticleTargets(data.id, data.indices, data.positions)
        break
      case 'step': {
        await backend.step(data.delta, data.options)
        const positions = [...bodies].map(id => [id, backend.getPositions(id).slice()])
        globalThis.postMessage(
          { positions, requestId: data.requestId, type: 'step' },
          positions.map(([, snapshot]) => snapshot.buffer),
        )
        break
      }
      case 'updateCollider':
        backend.updateCollider(data.id, data.descriptor)
        break
      case 'updateGrab':
        backend.updateGrab(data.id, data.position)
        break
      default:
        throw new Error(`Unknown CPU Worker command: ${data.type}`)
    }
  }
  catch (error) {
    respondWithError(error, data.requestId)
  }
}
