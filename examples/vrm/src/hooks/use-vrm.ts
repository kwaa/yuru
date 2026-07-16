import type { VRM, VRMLoaderPluginOptions } from '@pixiv/three-vrm'

import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm'
import { useLoader } from '@react-three/fiber'
import { useMemo } from 'react'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

const processed = new WeakSet<VRM>()

export const useVRM = (input: string, options?: VRMLoaderPluginOptions): VRM => {
  const gltf = useLoader(GLTFLoader, input, loader =>
    loader.register(parser => new VRMLoaderPlugin(parser, options)))

  return useMemo(() => {
    const vrm = gltf.userData.vrm as VRM
    if (processed.has(vrm))
      return vrm
    VRMUtils.rotateVRM0(vrm)
    VRMUtils.removeUnnecessaryVertices(vrm.scene)
    VRMUtils.combineSkeletons(vrm.scene)
    VRMUtils.combineMorphs(vrm)

    vrm.scene.traverse(object => object.frustumCulled = false)
    processed.add(vrm)

    return vrm
  }, [gltf])
}
