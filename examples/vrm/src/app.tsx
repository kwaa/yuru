import { Sky } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { BvhPhysicsBody, BvhPhysicsWorld, PrototypeBox } from '@react-three/viverse'
import { createXRStore, XR } from '@react-three/xr'

import { Avatar } from './components/avatar'
import { Player } from './components/player'

const store = createXRStore()

export const App = () => (
  <Canvas>
    <XR store={store}>
      <BvhPhysicsWorld>
        <Sky />
        <directionalLight castShadow intensity={1.2} position={[5, 10, 10]} />
        <ambientLight intensity={1} />
        <Player />
        <Avatar />
        <BvhPhysicsBody>
          <PrototypeBox position={[0, -0.5, 0]} scale={[10, 1, 15]} />
        </BvhPhysicsBody>
      </BvhPhysicsWorld>
    </XR>
  </Canvas>
)
