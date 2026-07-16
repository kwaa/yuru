import type { VRM } from '@pixiv/three-vrm'
import type {
  AttachYuruOptions,
  VRMClothCandidate,
  YuruController,
  YuruVRMStatus,
} from 'three-yuru/vrm'

import { useEffect, useRef, useState } from 'react'
import { attachYuru } from 'three-yuru/vrm'

import { useYuruWorld } from './index.js'

export interface UseYuruResult {
  candidates: readonly VRMClothCandidate[]
  controller: null | YuruController
  status: 'loading' | YuruVRMStatus
}

/** Conservatively discovers and attaches separate clothing meshes in a VRM. */
export const useYuru = (vrm: null | undefined | VRM, options: AttachYuruOptions = {}): UseYuruResult => {
  const world = useYuruWorld()
  const optionsRef = useRef(options)
  const [controller, setController] = useState<null | YuruController>(null)

  useEffect(() => {
    if (vrm == null)
      return
    const next = attachYuru(vrm, world, optionsRef.current)
    setController(next)
    return () => {
      next.dispose()
      setController(current => current === next ? null : current)
    }
  }, [vrm, world])

  return {
    candidates: controller?.candidates ?? [],
    controller,
    status: controller?.status ?? 'loading',
  }
}

export type { AttachYuruOptions, VRMClothCandidate, YuruController, YuruVRMStatus } from 'three-yuru/vrm'
