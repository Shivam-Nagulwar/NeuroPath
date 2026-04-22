# NeuroPath Frontend — Setup & Run Guide

## Prerequisites
- Node.js 18+ installed (https://nodejs.org)
- Backend must be running on localhost:8000

## Step 1 — Install dependencies

```bash
cd frontend
npm install
```

## Step 2 — Run the dev server

```bash
npm run dev
```

You should see:
```
  VITE v5.x  ready in 300ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://YOUR_IP:5173/    ← share this with your group
```

## Step 3 — Open in browser

- Your machine     : http://localhost:5173
- Same WiFi group  : http://YOUR_IP:5173

---

## File Structure

```
src/
├── App.jsx                  ← root, all state lives here
├── main.jsx                 ← ReactDOM entry
├── api/
│   └── neuropath.js         ← all API calls to FastAPI backend
├── components/
│   ├── Sidebar.jsx          ← step nav + triage card + buttons
│   ├── ChatWindow.jsx       ← scrollable message list
│   ├── MessageBubble.jsx    ← user/assistant bubbles
│   ├── MRIResultCard.jsx    ← gradcam + heatmap + probs card
│   ├── ReportPanel.jsx      ← full report + download button
│   └── InputBar.jsx         ← text input + MRI upload
└── styles/
    └── index.css            ← all styles (dark clinical theme)
```

## How it works

1. Vite proxies all `/api/*` requests to `http://localhost:8000`
   so no CORS issues during development.

2. `App.jsx` holds all state:
   - `messages[]` — visible chat thread
   - `apiHistory[]` — sent to backend for Gemini context
   - `triageSummary` — extracted and shown in sidebar
   - `mriDone` — enables Generate Report button

3. MRI results and reports render as special card components
   inline in the chat thread (not in separate tabs).
