import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { MOCK } from "./config.ts"

async function start() {
  if (MOCK) {
    const { worker } = await import("./mocks/browser.ts");
    await worker.start({ onUnhandledRequest: "bypass" });
  }
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
}

start();

