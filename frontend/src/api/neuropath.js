/**
 * NeuroPath API Client
 * All fetch() calls to the FastAPI backend in one place.
 * Vite proxy forwards /api/* → http://localhost:8000/api/*
 */

const BASE = '/api'

// ── Health ────────────────────────────────────────────────
export async function checkHealth() {
  const res = await fetch(`${BASE}/health`)
  return res.json()
}

// ── Step 1 — Triage Chat ──────────────────────────────────
/**
 * @param {string} message
 * @param {Array<{role: string, content: string}>} history
 */
export async function sendChatMessage(message, history = []) {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Chat request failed')
  }
  return res.json()
  // Returns: { reply, triage_summary, history }
}

// ── Step 2 — MRI Analysis ─────────────────────────────────
/**
 * @param {File} imageFile
 */
export async function analyseMRI(imageFile) {
  const formData = new FormData()
  formData.append('image', imageFile)

  const res = await fetch(`${BASE}/analyse-mri`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    // Gatekeeper rejection comes back as 422 with a structured detail object
    if (res.status === 422 && err.detail && err.detail.type === 'invalid_image') {
      const gatekeeperError = new Error(err.detail.message)
      gatekeeperError.isGatekeeperRejection = true
      gatekeeperError.detail = err.detail
      throw gatekeeperError
    }
    throw new Error(err.detail || 'MRI analysis failed')
  }
  return res.json()
  /*
  Returns: {
    pred_class, pred_label, confidence, description,
    risk_flag, probabilities, gradcam_b64, heatmap_b64, correlation
  }
  */
}

// ── Step 3 — Full Report ──────────────────────────────────
export async function generateReport() {
  const res = await fetch(`${BASE}/report`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Report generation failed')
  }
  return res.json()
  // Returns: { report, generated_at }
}

// ── Clear Session ─────────────────────────────────────────
export async function clearSession() {
  const res = await fetch(`${BASE}/clear`, { method: 'POST' })
  return res.json()
}