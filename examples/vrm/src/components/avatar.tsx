import type { ThreeEvent } from '@react-three/fiber'
import type { Group } from 'three'
import type { ThreeClothController } from 'three-yuru'
import type { GrabId } from 'yuru'

import { useFrame } from '@react-three/fiber'
import { useApplyXRSpaceMatrix, useXRInputSourceState } from '@react-three/xr'
import { useCallback, useEffect, useRef } from 'react'
import { useYuruWorld } from 'react-three-yuru'
import { useYuru } from 'react-three-yuru/vrm'
import { Vector3 } from 'three'

import AvatarSampleB from '../assets/AvatarSample_B.vrm?url'

import { useVRM } from '../hooks/use-vrm'

export interface AvatarAnalysis {
  candidate?: string
  status: 'loading' | 'needsConfiguration' | 'ready'
  triangles?: number
}

interface ControllerGrabberProps {
  cloth: readonly ThreeClothController[]
  handedness: XRHandedness
}

const useVolumeGrabs = (cloth: readonly ThreeClothController[]) => {
  const world = useYuruWorld()
  const grabsRef = useRef<GrabId[]>([])

  const release = useCallback(() => {
    for (const grab of grabsRef.current)
      world.core.removeGrab(grab)
    grabsRef.current = []
  }, [world])

  const update = useCallback((position: Vector3, radius: number) => {
    if (grabsRef.current.length === 0) {
      grabsRef.current = cloth.map(controller => world.core.addGrab({
        body: controller.body,
        compliance: 1e-8,
        position,
        radius,
      }))
      return
    }
    for (const grab of grabsRef.current)
      world.core.updateGrab(grab, position)
  }, [cloth, world])

  useEffect(() => release, [cloth, release])
  return { release, update }
}

type HandGrabberProps = ControllerGrabberProps

const HandGrabber = ({ cloth, handedness }: HandGrabberProps) => {
  const state = useXRInputSourceState('hand', handedness)
  const thumbSpace = state?.inputSource.hand?.get('thumb-tip')
  const indexSpace = state?.inputSource.hand?.get('index-finger-tip')
  const thumbRef = useRef<Group>(null)
  const indexRef = useRef<Group>(null)
  const thumbPointRef = useRef(new Vector3())
  const indexPointRef = useRef(new Vector3())
  const pinchPointRef = useRef(new Vector3())
  const pinchingRef = useRef(false)
  const { release, update } = useVolumeGrabs(cloth)

  const updatePinch = useCallback(() => {
    const thumb = thumbRef.current
    const index = indexRef.current
    if (thumb?.visible !== true || index?.visible !== true) {
      pinchingRef.current = false
      release()
      return
    }

    thumb.getWorldPosition(thumbPointRef.current)
    index.getWorldPosition(indexPointRef.current)
    const distance = thumbPointRef.current.distanceTo(indexPointRef.current)
    if (!pinchingRef.current) {
      if (distance > 0.03)
        return
      pinchingRef.current = true
    }
    else if (distance >= 0.045) {
      pinchingRef.current = false
      release()
      return
    }

    pinchPointRef.current
      .copy(thumbPointRef.current)
      .add(indexPointRef.current)
      .multiplyScalar(0.5)
    update(pinchPointRef.current, 0.08)
  }, [release, update])

  useApplyXRSpaceMatrix(thumbRef, thumbSpace)
  useApplyXRSpaceMatrix(indexRef, indexSpace, updatePinch)

  return (
    <>
      <group matrixAutoUpdate={false} ref={thumbRef} />
      <group matrixAutoUpdate={false} ref={indexRef} />
    </>
  )
}

const ControllerGrabber = ({ cloth, handedness }: ControllerGrabberProps) => {
  const state = useXRInputSourceState('controller', handedness)
  const handState = useXRInputSourceState('hand', handedness)
  const pointRef = useRef(new Vector3())
  const { release, update } = useVolumeGrabs(cloth)

  useFrame(() => {
    if (handState != null || state?.object == null) {
      release()
      return
    }
    const pressed = state.gamepad['xr-standard-squeeze']?.state === 'pressed'
      || state.gamepad['xr-standard-trigger']?.state === 'pressed'
    state.object.getWorldPosition(pointRef.current)
    if (!pressed) {
      release()
      return
    }
    update(pointRef.current, 0.12)
  }, -2)

  return null
}

interface AvatarProps {
  onAnalysis?: (analysis: AvatarAnalysis) => void
  url?: string
}

export const Avatar = ({ onAnalysis, url = AvatarSampleB }: AvatarProps) => {
  const vrm = useVRM(url, { autoUpdateHumanBones: true })
  const world = useYuruWorld()
  const yuru = useYuru(vrm)
  const pointerGrabRef = useRef<GrabId | null>(null)

  useFrame((_state, delta) => vrm.update(delta), -3)

  useEffect(() => {
    const candidate = yuru.candidates[0]
    onAnalysis?.({
      candidate: candidate?.mesh.name,
      status: yuru.status,
      triangles: candidate?.triangles?.length,
    })
  }, [onAnalysis, yuru.candidates, yuru.status])

  useEffect(() => () => {
    if (pointerGrabRef.current == null)
      return
    world.core.removeGrab(pointerGrabRef.current)
  }, [world])

  const findCloth = (object: object): ThreeClothController | undefined =>
    yuru.controller?.cloth.find(controller => controller.mesh === object)

  const onPointerDown = (event: ThreeEvent<PointerEvent>) => {
    const controller = findCloth(event.object)
    if (controller == null)
      return
    event.stopPropagation()
    if (pointerGrabRef.current != null)
      world.core.removeGrab(pointerGrabRef.current)
    pointerGrabRef.current = world.core.addGrab({
      body: controller.body,
      compliance: 1e-8,
      position: event.point,
      radius: 0.1,
    })
  }

  const onPointerMove = (event: ThreeEvent<PointerEvent>) => {
    if (pointerGrabRef.current == null)
      return
    world.core.updateGrab(pointerGrabRef.current, event.point)
  }

  const onPointerUp = () => {
    if (pointerGrabRef.current == null)
      return
    world.core.removeGrab(pointerGrabRef.current)
    pointerGrabRef.current = null
  }

  return (
    <>
      <primitive
        dispose={null}
        object={vrm.scene}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        position={[0, 0, -2]}
      />
      <HandGrabber cloth={yuru.controller?.cloth ?? []} handedness="left" />
      <HandGrabber cloth={yuru.controller?.cloth ?? []} handedness="right" />
      <ControllerGrabber cloth={yuru.controller?.cloth ?? []} handedness="left" />
      <ControllerGrabber cloth={yuru.controller?.cloth ?? []} handedness="right" />
    </>
  )
}
