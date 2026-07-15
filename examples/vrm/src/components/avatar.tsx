import AvatarSampleB from '../assets/AvatarSample_B.vrm?url'

import { useVRM } from '../hooks/use-vrm'

export const Avatar = () => {
  const vrm = useVRM(AvatarSampleB, { autoUpdateHumanBones: true })

  return (
    <primitive
      dispose={null}
      object={vrm.scene}
      position={[0, 0, -2]}
    />
  )
}
