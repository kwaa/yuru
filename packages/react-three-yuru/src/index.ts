import type { PropsWithChildren } from 'react'
import type { Mesh } from 'three'
import type { ThreeClothController, ThreeClothOptions, ThreeYuruWorld } from 'three-yuru'
import type { ClothWorldOptions } from 'yuru'

import { useFrame } from '@react-three/fiber'
import { createContext, createElement, use, useEffect, useRef, useState } from 'react'
import { createThreeYuruWorld } from 'three-yuru'

const YuruContext = createContext<null | ThreeYuruWorld>(null)

export interface YuruProviderProps extends PropsWithChildren {
  /** Read once when the provider creates its world. */
  options?: ClothWorldOptions
  /** Runs after animation/physics hooks with lower priorities and before rendering. */
  updatePriority?: number
  /** Supply an existing world when sharing it with non-React code. */
  world?: ThreeYuruWorld
}

export const YuruProvider = ({
  children,
  options,
  updatePriority = -1,
  world: suppliedWorld,
}: YuruProviderProps) => {
  const [world] = useState(() => suppliedWorld ?? createThreeYuruWorld(options))
  const lifecycleRef = useRef(0)
  const ownsWorld = suppliedWorld == null

  useFrame((_state, delta) => {
    void world.update(delta)
  }, updatePriority)

  useEffect(() => {
    lifecycleRef.current++
    return () => {
      const version = ++lifecycleRef.current
      queueMicrotask(() => {
        // React StrictMode immediately re-runs effects; only a real unmount
        // leaves this cleanup as the latest lifecycle event.
        // eslint-disable-next-line react/exhaustive-deps -- The latest ref value distinguishes a remount from an unmount.
        if (!ownsWorld || lifecycleRef.current !== version)
          return
        world.dispose()
      })
    }
  }, [ownsWorld, world])

  return createElement(YuruContext.Provider, { value: world }, children)
}

/** Alias that reads naturally beside Canvas and Physics providers. */
export const YuruWorld = YuruProvider

export const useYuruWorld = (): ThreeYuruWorld => {
  const world = use(YuruContext)
  if (world == null)
    throw new Error('useYuruWorld must be used inside <YuruProvider>')
  return world
}

/**
 * Attaches a Three mesh for the lifetime of the component. Options are read
 * when the mesh is attached; remount the component to change topology options.
 */
export const useCloth = (
  mesh: Mesh | null | undefined,
  options: ThreeClothOptions = {},
): null | ThreeClothController => {
  const world = useYuruWorld()
  const optionsRef = useRef(options)
  const [controller, setController] = useState<null | ThreeClothController>(null)

  useEffect(() => {
    if (mesh == null)
      return
    const next = world.attachCloth(mesh, optionsRef.current)
    setController(next)
    return () => {
      next.dispose()
      setController(current => current === next ? null : current)
    }
  }, [mesh, world])

  return controller
}

export type { ThreeClothController, ThreeClothOptions, ThreeYuruWorld } from 'three-yuru'
export type { ClothWorldOptions } from 'yuru'
