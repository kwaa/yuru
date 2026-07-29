# yuru-wasm

Optional Rust/WASM solver acceleration for [`yuru`](../yuru). It is a separate
package so WASM toolchain, SIMD, Worker, and deployment requirements do not
affect the portable core package.

## Build

Provide `cargo` and `rustc`, then enter the Nix development shell for
`wasm-pack`, `wasm-bindgen`, Binaryen, and the LLVM tools. Run:

```sh
pnpm -F yuru-wasm wasm:build
pnpm -F yuru-wasm build
```

The crate is always compiled with `#![no_std]`, uses `alloc` backed by
`dlmalloc`, and enables WASM SIMD. `tsdown` and `rolldown-plugin-wasm` bundle the
generated module and its WASM asset into `dist`; there is no separate JavaScript
build script.

## Usage

```ts
import { createClothWorld } from 'yuru'
import { createWasmBackend } from 'yuru-wasm'

const backend = await createWasmBackend()
const world = createClothWorld({ backend })
```

In browsers, backend creation starts a dedicated module Worker by default.
When `SharedArrayBuffer` and `Atomics.waitAsync` are available, particle
positions are shared with that Worker and a step completes through an atomic
sequence signal; otherwise the same Worker falls back to transferable
`postMessage` snapshots. Use `{ worker: false }` to run inline.

The shared path requires a
[cross-origin isolated](https://developer.mozilla.org/docs/Web/API/Window/crossOriginIsolated)
page. Production responses should include:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The included Vite example sets both headers for its development and preview
servers. Backend creation is explicit and rejects when WebAssembly or the
generated module is unavailable.

An already loaded module may be injected for CSP-restricted deployments or
tests:

```ts
const backend = await createWasmBackend({ module })
```

Injected modules run inline because module namespace objects are not
structured-cloneable.

The WASM solver retains cooked topology and material data across frames. A
single structural call performs prediction, speed limiting, geodesic tethers,
distance constraints, and area constraints. Vertex/triangle and edge/edge
self/inter-cloth collision broad phases and contact solving also run inside
WASM. Animated targets, force sampling, external collider handling, and final
render binding remain in the shared TypeScript pipeline.
