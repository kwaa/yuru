# yuru

Renderer-independent Small Steps XPBD cloth simulation for TypeScript.

`yuru` contains no Three.js, React, Rapier, WebGPU, or WASM types. Vector and
quaternion inputs accept tuples as well as structurally compatible objects such
as `{ x, y, z }` and `{ x, y, z, w }`.

## Install

```sh
pnpm add yuru
```

## Usage

```ts
import { createClothWorld } from 'yuru'

const world = createClothWorld({ quality: 'high', speedLimit: 'automatic' })
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
const positions = world.getPositions(body)
```

Omitting `backend` always creates `CPUBackend`. In browsers it owns a dedicated
module Worker; non-browser runtimes fall back to the same typed-array solver on
the calling thread. To use another implementation, pass a `ClothBackend` object
explicitly:

```ts
const world = createClothWorld({ backend })
```

There is no string-based backend selector and no implicit backend switch.

## Current solver features

- Fixed-step accumulation with bounded catch-up.
- Stretch, bend-distance, and area constraints.
- Geodesic long-range tethers from dynamic particles to connected pins. They
  default on when pins exist and can be disabled with `tethers: false`.
- Vertex/triangle and edge/edge self- and inter-cloth collision.
- Ordered garment layers with an optional axis-projected contact normal for
  inner/body, simulated garment, and outer-garment relationships.
- Density-aware particle displacement limiting and bounded initial-overlap
  correction for robust self- and inter-layer collision.
- Connected-particle velocity damping plus bounded inelastic contact friction.
- Sphere, capsule, plane, rounded-box, and triangle-mesh colliders.
- Continuous particle collision against moving spheres and capsules in the
  high-quality preset. Other collider and cloth-contact paths remain discrete.
- Friction, wind/aerodynamic forces, configurable force fields, and volume grabs.
- Low, medium, and high quality presets.

## Development

```sh
pnpm -F yuru build
pnpm vitest run packages/yuru/test
```
