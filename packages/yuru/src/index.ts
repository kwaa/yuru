export { CPUBackend } from './cpu-backend.js'
export type { CPUBackendOptions, CPUWorkerLike } from './cpu-backend.js'
/** Advanced entry point used by dedicated numeric-backend Workers. */
export { CPUSolverBackend } from './cpu-solver-backend.js'
export {
  DEFAULT_CLOTH_MATERIAL,
  DEFAULT_COLLISION_FILTER,
  QUALITY_PRESETS,
} from './types.js'
export type * from './types.js'
export { ClothWorld, createClothWorld } from './world.js'
