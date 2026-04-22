# NeuroPath Backend — Setup & Run Guide

## Folder structure expected
```
backend/
├── main.py
├── cnn_engine.py
├── gemini_engine.py
├── config.py
├── requirements.txt
├── .env
├── .env.example
└── models/
    └── neuropath_xception_best.h5   ← copy your model here from Google Drive
```

---

## Step 1 — Copy your model file

Download `neuropath_xception_best.h5` from Google Drive and place it inside:
```
backend/models/neuropath_xception_best.h5
```

---

## Step 2 — Create a virtual environment

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

---

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Set up your Gemini API key

1. Copy the `.env.example` file to create a `.env` file:
   ```bash
   copy .env.example .env
   ```

2. Open the `.env` file and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_key_here
   ```

3. Save the file. **Do not commit this file to version control.**

---

## Step 5 — Run the server

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 --env-file .env
```

You should see:
```
✓  Model loaded from models/neuropath_xception_best.h5
✓  Grad-CAM engine ready
✓  Gemini triage model ready
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## API Endpoints

| Method | URL | Purpose |
|--------|-----|---------|
| GET | /api/health | Check server is alive |
| POST | /api/chat | Send chat message |
| POST | /api/analyse-mri | Upload MRI image |
| POST | /api/report | Generate full report |
| POST | /api/clear | Clear session |

Interactive docs available at: http://localhost:8000/docs
