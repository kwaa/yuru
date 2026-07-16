import type { RapierColliderLike } from 'three-yuru/rapier'

import { useFrame } from '@react-three/fiber'
import { useEffect, useState } from 'react'
import { RapierColliderBridge } from 'three-yuru/rapier'

import { useYuruWorld } from './index.js'

/** Mirrors an existing Rapier collider into Yuru before each cloth step. */
export const useRapierCollider = (
  collider: null | RapierColliderLike | undefined,
  updatePriority = -2,
): null | RapierColliderBridge => {
  const world = useYuruWorld()
  const [bridge, setBridge] = useState<null | RapierColliderBridge>(null)

  useEffect(() => {
    if (collider == null)
      return
    const next = new RapierColliderBridge(world, collider)
    // eslint-disable-next-line react/set-state-in-effect -- The bridge exists only after the resource effect attaches it.
    setBridge(next)
    return () => {
      next.dispose()
      setBridge(current => current === next ? null : current)
    }
  }, [collider, world])

  useFrame(() => bridge?.update(), updatePriority)
  return bridge
}

export { colliderFromRapier, RapierColliderBridge } from 'three-yuru/rapier'
export type { RapierColliderLike, RapierRotation, RapierVector } from 'three-yuru/rapier'
