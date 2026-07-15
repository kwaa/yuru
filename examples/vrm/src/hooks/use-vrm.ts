import type { VRM, VRMLoaderPluginOptions } from '@pixiv/three-vrm'

import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm'
import { useLoader } from '@react-three/fiber'
import { useMemo } from 'react'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

export const useVRM = (input: string, options?: VRMLoaderPluginOptions): VRM => {
  const gltf = useLoader(GLTFLoader, input, loader =>
    loader.register(parser => new VRMLoaderPlugin(parser, options)))

  return useMemo(() => {
    const vrm = gltf.userData.vrm as VRM
    VRMUtils.rotateVRM0(vrm)
    VRMUtils.removeUnnecessaryVertices(vrm.scene)
    VRMUtils.combineSkeletons(vrm.scene)
    VRMUtils.combineMorphs(vrm)

    vrm.scene.traverse(object => object.frustumCulled = false)

    return vrm
  }, [gltf])
}
