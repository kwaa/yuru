# Yuru

Yuru is a TypeScript cloth simulation workspace aimed at character clothing in
VRM/MMD-style applications. The core is renderer-independent, defaults to a
Worker-backed CPU backend, and accepts Three-compatible vector/quaternion shapes without
importing or exposing Three types.

The current milestone establishes the package boundaries, Small Steps XPBD
CPU solver, self-collision path, VRM discovery, and interactive example
needed to measure and improve simulation quality.

## Packages

| Package | Purpose |
| --- | --- |
| `yuru` | Renderer-independent cloth world, Worker-backed CPU backend, constraints, forces, grabs, and collision shapes. |
| `three-yuru` | Three mesh binding, simulation proxies, TSL experiment, Rapier collider adapter, and pixiv/three-vrm integration. |
| `react-three-yuru` | React Three Fiber provider and hooks for cloth, VRM, and Rapier. |
| `yuru-wasm` | Separate optional Rust/WASM integration package. It is not required by the default CPU or TSL build. |
| `@yurujs/example-vrm` | Desktop/WebXR demo using the official `AvatarSample_B.vrm`. |

## Core API

The backend is an object, never a string selector. Omitting it always selects
the CPU implementation.

```ts
import { createClothWorld } from 'yuru'

const world = createClothWorld({
  gravity: { x: 0, y: -9.81, z: 0 }, // structurally compatible with THREE.Vector3
  quality: 'high',
})

const body = world.addBody({
  mesh: {
    indices: new Uint16Array([0, 1, 2]),
    inverseMasses: new Float32Array([0, 1, 1]),
    positions: new Float32Array([0, 1, 0, 1, 1, 0, 0, 0, 0]),
  },
  selfCollision: true,
  tethers: true,
})

await world.step(1 / 60)
world.getPositions(body)
```

The CPU backend currently includes stretch, bend-distance, area, and geodesic
tether constraints,
fixed-step accumulation, vertex/triangle and edge/edge self/inter-cloth
collisions, sphere/capsule/plane/rounded-box/triangle-mesh colliders, friction,
wind/aerodynamic forces, volume grabs, and continuous particle collision against
moving spheres and capsules in the high-quality preset.

## Three and WebGPU

```ts
import { createThreeYuruWorld, probeTSLBackend } from 'three-yuru'
import { createTSLBackend } from 'three-yuru/tsl'

const probe = probeTSLBackend(renderer)
const backend = probe.supported ? createTSLBackend(renderer) : undefined
const world = createThreeYuruWorld({ backend }) // undefined means CPU
const cloth = world.attachCloth(mesh, { pin: 'top', selfCollision: true })
```

`three-yuru/tsl` is a real TSL storage-buffer prediction kernel, but it currently
reads positions back for CPU XPBD constraints and collision handling. It is an
experimental validation path, not yet the desired all-GPU solver. Backend
selection stays explicit so an application can report this distinction rather
than silently changing behavior.

For a denser render mesh, pass a lower-resolution `simulationMesh`; the binding
maps display vertices to the proxy while preserving their rest offsets.

## VRM

```ts
import { attachYuru } from 'three-yuru/vrm'

const controller = attachYuru(vrm, world)
console.log(controller.status, controller.candidates)
```

Discovery handles both separately named clothing meshes and clothing embedded
inside a merged `Body` mesh. For merged meshes it finds vertices weighted to
secondary skirt/coat/clothing bones, extracts only matching triangles, keeps the
rest of the original skinned mesh visible, and creates height-scaled skeletal
capsule colliders for a narrow vertical body center and the individual legs. If
The extracted skirt is tethered to its animated waist boundary by mesh
connectivity rather than spatial proximity. If confidence remains low it returns
`needsConfiguration` instead of simulating a body, face, or hair by accident.

The Vitest suite loads the included official `AvatarSample_B.vrm` and verifies
that its merged skirt is detected, extracted, attached, and restored on dispose.

## React Three Fiber

```tsx
import { YuruProvider } from 'react-three-yuru'
import { useYuru } from 'react-three-yuru/vrm'

export const Avatar = ({ vrm }) => {
  const yuru = useYuru(vrm)
  return <primitive object={vrm.scene} userData={{ yuruStatus: yuru.status }} />
}

export const Scene = () => (
  <Canvas>
    <YuruProvider options={{ quality: 'high' }}>
      <Avatar vrm={vrm} />
    </YuruProvider>
  </Canvas>
)
```

The example supports loading local VRM files, desktop pointer grabs, hand
tracking pinch grabs, and controller squeeze/trigger fallback grabs.

## Development

```sh
pnpm install
pnpm check
pnpm dev
```

`pnpm check` runs ESLint, TypeScript, Vitest, and package/example builds. The
separate Rust toolchain command is `pnpm wasm:build`; it is intentionally not
required for CPU, TSL, Three, React, or example development.

## License

[MIT](./LICENSE.md)
