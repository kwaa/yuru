import type { AvatarAnalysis } from './components/avatar'

import { Sky } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { BvhPhysicsBody, BvhPhysicsWorld, PrototypeBox } from '@react-three/viverse'
import { createXRStore, XR } from '@react-three/xr'
import { Suspense, use, useEffect, useRef, useState } from 'react'
import { YuruProvider } from 'react-three-yuru'
import { createWasmBackend } from 'yuru-wasm'

import AvatarSampleB from './assets/AvatarSample_B.vrm?url'

import { Avatar } from './components/avatar'
import { Player } from './components/player'

const store = createXRStore({ handTracking: true })
const backendPromise = createWasmBackend()

export const App = () => {
  const backend = use(backendPromise)
  const [avatarUrl, setAvatarUrl] = useState(AvatarSampleB)
  const [analysis, setAnalysis] = useState<AvatarAnalysis>({ status: 'loading' })
  const objectUrlRef = useRef<null | string>(null)

  useEffect(() => () => {
    if (objectUrlRef.current == null)
      return
    URL.revokeObjectURL(objectUrlRef.current)
  }, [])

  const loadAvatar = (file?: File) => {
    if (file == null)
      return
    if (objectUrlRef.current != null)
      URL.revokeObjectURL(objectUrlRef.current)
    objectUrlRef.current = URL.createObjectURL(file)
    setAnalysis({ status: 'loading' })
    setAvatarUrl(objectUrlRef.current)
  }

  const statusText = analysis.status === 'ready'
    ? `Ready · ${analysis.triangles ?? 0} cloth triangles`
    : analysis.status === 'needsConfiguration'
      ? 'No safe cloth region found'
      : 'Analyzing VRM…'
  const candidateLabel = analysis.candidate == null ? null : <small>{analysis.candidate}</small>

  return (
    <>
      <Canvas shadows>
        <XR store={store}>
          <YuruProvider options={{ backend, quality: 'medium' }}>
            <BvhPhysicsWorld>
              <Sky />
              <directionalLight castShadow intensity={1.2} position={[5, 10, 10]} />
              <ambientLight intensity={1} />
              <Player />
              <Suspense fallback={null}>
                <Avatar key={avatarUrl} onAnalysis={setAnalysis} url={avatarUrl} />
              </Suspense>
              <BvhPhysicsBody>
                <PrototypeBox position={[0, -0.5, 0]} scale={[10, 1, 15]} />
              </BvhPhysicsBody>
            </BvhPhysicsWorld>
          </YuruProvider>
        </XR>
      </Canvas>
      <aside className="controls">
        <div className="brand">Yuru · XPBD cloth</div>
        <div className={`status status-${analysis.status}`}>
          {statusText}
        </div>
        {candidateLabel}
        <p>Pinch the skirt with tracked hands, or use a controller squeeze/trigger as fallback.</p>
        <div className="actions">
          <button data-test-id="enter-vr" onClick={() => void store.enterVR()} type="button">Enter VR</button>
          <label>
            Load VRM
            <input accept=".vrm,model/gltf-binary" data-test-id="load-vrm" onChange={event => loadAvatar(event.target.files?.[0])} type="file" />
          </label>
        </div>
      </aside>
    </>
  )
}
