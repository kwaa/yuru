# three-yuru

Three.js bindings for [`yuru`](../yuru), including VRM discovery, simulation
proxy binding, collider conversion, Rapier adaptation, and an experimental TSL
prediction backend.

## Install

```sh
pnpm add yuru three three-yuru
```

## Mesh binding

```ts
import { createThreeYuruWorld } from 'three-yuru'

const world = createThreeYuruWorld({ quality: 'high' })
const cloth = world.attachCloth(mesh, {
  pin: 'top',
  selfCollision: true,
})

await world.update(delta)
```

Use `simulationMesh` to drive a dense display mesh with a lower-resolution
proxy. The display binding preserves each visual vertex's initial offset from
its nearest simulation particle.

Use `attachKinematicClothCollider()` for a skinned or transformed collision
surface that should participate in cloth inter-collision without being
simulated or rendered by Yuru.

## VRM

```ts
import { attachYuru } from 'three-yuru/vrm'

const controller = attachYuru(vrm, world)
console.log(controller.status, controller.candidates)
```

Detection supports separate clothing meshes and clothing embedded in a merged
skinned body mesh. Merged regions are selected from secondary clothing-bone
weights and extracted without hiding the remaining body. The automatic path
welds render-only UV/material seam splits into an unrendered simulation proxy,
pins the actual upper boundary loop, and treats a nearby outer garment island
as a separately skinned kinematic layer. Ordered, gravity-axis-projected contact
keeps the dynamic skirt below the coat without lifting its hem, preserves the
authored rest gap, and uses low interface friction so pleats can slide past the
coat edge. Individual leg capsules are fitted inside the rest-pose clearance;
a narrow vertical center proxy prevents panels collapsing between the legs,
while the broad horizontal pelvis proxy remains omitted. Low-confidence
results return `needsConfiguration` instead of simulating an unsafe candidate.
The automatic VRM path also enables topology-aware tethers from skirt particles
to the animated waist boundary to limit accumulated panel stretch without
resisting folds that move closer to the waist. It derives per-particle motion
limits from simulation-mesh spacing and updates their target centers from the
skinned proxy each frame, preventing side and rear panels from collapsing into
the legs while preserving local cloth motion.

## TSL/WebGPU

```ts
import { createThreeYuruWorld, probeTSLBackend } from 'three-yuru'
import { createTSLBackend } from 'three-yuru/tsl'

const backend = probeTSLBackend(renderer).supported
  ? createTSLBackend(renderer)
  : undefined
const world = createThreeYuruWorld({ backend })
```

The current TSL implementation predicts particles in WebGPU storage buffers,
then reads them back for CPU XPBD constraints and collision handling. Backend
selection remains explicit so applications can distinguish this experimental
path from the default CPU backend.

## Collider adapters

- `three-yuru`: `Sphere`, `Plane`, `Box3`, capsule-like objects, and
  `BufferGeometry` conversion.
- `three-yuru/rapier`: structural Rapier collider conversion and a live bridge.

## Development

```sh
pnpm -F three-yuru build
pnpm vitest run packages/three-yuru/test
```
