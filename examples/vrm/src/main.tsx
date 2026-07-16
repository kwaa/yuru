import { StrictMode, Suspense } from 'react'
import { createRoot } from 'react-dom/client'

import { App } from './app'

import './main.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Suspense fallback={<div className="loading">Loading WASM…</div>}>
      <App />
    </Suspense>
  </StrictMode>,
)
