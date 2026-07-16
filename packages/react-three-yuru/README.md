# react-three-yuru

React Three Fiber bindings for [`three-yuru`](../three-yuru) and
[`yuru`](../yuru).

## Install

```sh
pnpm add react three @react-three/fiber yuru three-yuru react-three-yuru
```

## Provider and cloth hook

`YuruProvider` must be rendered inside `<Canvas>` because it advances the shared
world with `useFrame` before rendering.

```tsx
import { useCloth, YuruProvider } from 'react-three-yuru'

export const Cloth = ({ mesh }) => {
  useCloth(mesh, { pin: 'top', selfCollision: true })
  return null
}

export const Scene = () => (
  <Canvas>
    <YuruProvider options={{ quality: 'high' }}>
      <Cloth mesh={mesh} />
    </YuruProvider>
  </Canvas>
)
```

Supply `world` to `YuruProvider` when React and non-React code need to share an
existing `ThreeYuruWorld`.

## VRM

```tsx
import { useYuru } from 'react-three-yuru/vrm'

export const Avatar = ({ vrm }) => {
  const { candidates, status } = useYuru(vrm)
  return <primitive object={vrm.scene} userData={{ candidates, status }} />
}
```

`useYuru` performs the same conservative separate/merged clothing discovery as
`three-yuru/vrm` and disposes extracted geometry and colliders with the
component.

## Rapier

```tsx
import { useRapierCollider } from 'react-three-yuru/rapier'

export const RapierBridge = ({ collider }) => {
  useRapierCollider(collider)
  return null
}
```

The hook mirrors an existing structural Rapier collider into the Yuru world
before each cloth step.

## Development

```sh
pnpm -F react-three-yuru build
```
