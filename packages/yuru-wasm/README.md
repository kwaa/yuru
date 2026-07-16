# yuru-wasm

Optional Rust/WASM numeric acceleration for [`yuru`](../yuru). It is a separate
package so WASM toolchain, SIMD, and deployment requirements do not affect the
portable core package.

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

Backend creation is explicit and rejects when WebAssembly or the generated
module is unavailable.

An already loaded module may be injected for CSP-restricted deployments or
tests:

```ts
const backend = await createWasmBackend({ module })
```

The current kernel uses WebAssembly SIMD for particle prediction. XPBD
constraints and collisions remain in the shared typed-array CPU solver.
