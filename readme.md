<div align="center">

# 🧠 NeuroPath AI

### Integrated Neurological Symptom Triage & Explainable Brain Tumour Detection

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat&logo=google&logoColor=white)](https://deepmind.google/gemini)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

<br/>

![NeuroPath Banner](https://via.placeholder.com/900x300/0a0a0a/00d4ff?text=NeuroPath+AI+%E2%80%94+Neurological+Triage+%2B+MRI+Analysis)

<br/>

> **⚠️ Medical Disclaimer:** NeuroPath is an AI-powered screening and research tool only.
> It does **not** constitute a medical diagnosis. Always consult a qualified
> radiologist or neurologist for clinical decisions.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [How to Use](#-how-to-use)
- [API Reference](#-api-reference)
- [Screenshots](#-screenshots)
- [Team](#-team)
- [License](#-license)

---

## 🔍 Overview

**NeuroPath** is an end-to-end AI diagnostic assistant for neurological conditions. It combines a **Gemini-powered conversational triage chatbot** with a **deep learning MRI classifier** and **Grad-CAM explainability** into a single, cohesive web application.

The system guides a user through three steps:

1. **Symptom Triage** — An AI chatbot interviews the user about their neurological symptoms, assesses risk level (LOW / MEDIUM / HIGH), and decides whether an MRI scan is warranted.
2. **MRI Analysis** — A fine-tuned Xception CNN classifies the uploaded brain MRI into one of four categories, with Grad-CAM heatmaps highlighting the regions the model focused on.
3. **Full Report** — Gemini synthesises the triage conversation, CNN prediction, and symptom-location correlation into a plain-language diagnostic report the patient can download.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🩺 **Conversational Triage** | Gemini 2.5 Flash chatbot that interviews patients, assesses neurological risk, and generates a structured triage summary |
| 🔬 **MRI Classification** | Xception CNN trained on 7,000+ MRI scans classifies Glioma, Meningioma, Pituitary Tumour, or No Tumour |
| 🌡️ **Grad-CAM Explainability** | Visual heatmaps overlaid on the MRI showing exactly which regions influenced the AI's prediction |
| 🧩 **Symptom-Location Correlation** | Rule-based engine maps reported symptoms to brain lobes affected by the predicted tumour type |
| 📄 **Integrated Report** | Gemini writes a warm, plain-language report combining all three steps, downloadable as a text file |
| 🔒 **Scope Enforcement** | The triage AI refuses non-neurological questions and redirects to appropriate specialists |
| ⚡ **Modern Web UI** | ChatGPT-style React interface — chat, upload MRI, and view results all in one thread |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        FRONTEND                              │
│              React + Vite  (localhost:5173)                  │
│                                                              │
│  ┌──────────┐  ┌─────────────┐  ┌──────────┐  ┌─────────┐  │
│  │ Sidebar  │  │ ChatWindow  │  │MRIResult │  │ Report  │  │
│  │ Step Nav │  │  Messages   │  │   Card   │  │  Panel  │  │
│  │ Triage   │  │  Bubbles    │  │ GradCAM  │  │Download │  │
│  │ Summary  │  │             │  │ Heatmap  │  │         │  │
│  └──────────┘  └─────────────┘  └──────────┘  └─────────┘  │
└────────────────────────┬─────────────────────────────────────┘
                         │  HTTP / REST  (Vite proxy)
┌────────────────────────▼─────────────────────────────────────┐
│                        BACKEND                               │
│              FastAPI + Uvicorn  (localhost:8000)             │
│                                                              │
│  ┌──────────────────┐     ┌──────────────────────────────┐  │
│  │  gemini_engine   │     │        cnn_engine            │  │
│  │                  │     │                              │  │
│  │ • Triage chatbot │     │ • Xception CNN (299×299)     │  │
│  │ • System prompt  │     │ • Grad-CAM engine            │  │
│  │ • Risk scoring   │     │ • Symptom-lobe correlation   │  │
│  │ • Report writer  │     │ • base64 image encoding      │  │
│  └────────┬─────────┘     └──────────────┬───────────────┘  │
│           │                              │                   │
│           ▼                              ▼                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    main.py (FastAPI)                │    │
│  │  POST /api/chat  │  POST /api/analyse-mri           │    │
│  │  POST /api/report│  POST /api/clear                 │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
           │                              │
           ▼                              ▼
  Google Gemini API            .h5 Model File (local)
  (gemini-2.5-flash)         neuropath_xception_best.h5
```

---

## 🛠️ Tech Stack

### Backend
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| FastAPI | 0.115 | REST API server |
| Uvicorn | 0.30 | ASGI web server |
| TensorFlow / Keras | 2.16 | CNN model inference |
| Xception | Transfer Learning | Brain tumour classification |
| OpenCV | 4.10 | Grad-CAM heatmap rendering |
| Google Generative AI | 0.8 | Gemini 2.5 Flash triage + report |
| Pillow | 10.4 | Image preprocessing |

### Frontend
| Technology | Version | Purpose |
|---|---|---|
| React | 18.3 | UI framework |
| Vite | 5.4 | Build tool + dev server |
| Lucide React | 0.383 | Icon library |
| CSS Variables | — | Dark clinical theming |

### AI Models
| Model | Task | Details |
|---|---|---|
| Xception CNN | MRI Classification | Fine-tuned on Brain Tumour MRI Dataset, 299×299 input |
| Gemini 2.5 Flash Lite | Symptom Triage | Custom system prompt, neurological scope enforcement |
| Gemini 2.5 Flash | Report Generation | Separate instance, no triage system instruction |
| Grad-CAM | Explainability | Target layer: `block14_sepconv2_act` |

---

## 📊 Model Performance

The CNN was trained and evaluated on the [Brain Tumour MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) with an 80/20 train-validation split.

### Test Set Results

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| Glioma | 0.958 | 0.775 | 0.866 |
| Meningioma | 0.863 | 0.957 | 0.907 |
| No Tumour | 0.929 | 1.000 | 0.963 |
| Pituitary | 0.944 | 0.998 | 0.971 |
| **Weighted Avg** | **0.924** | **0.933** | **0.927** |

> ⚠️ **Known Limitation:** Glioma recall is 77.5%. Any MRI where glioma is clinically suspected must be confirmed by a qualified radiologist, regardless of the AI's output.

### Training Configuration

| Parameter | Value |
|---|---|
| Base Model | Xception (ImageNet weights) |
| Input Size | 299 × 299 × 3 |
| Batch Size | 32 |
| Phase 1 Epochs | 20 |
| Phase 2 Fine-tuning | 10 epochs, last 30 layers unfrozen |
| Optimizer | Adam (LR: 0.001 → 1e-5) |
| Classes | 4 (Glioma, Meningioma, No Tumour, Pituitary) |

---

## 📁 Project Structure

```
NeuroPath/
│
├── backend/                          # FastAPI Python server
│   ├── main.py                       # API endpoints + session state
│   ├── cnn_engine.py                 # CNN inference + Grad-CAM engine
│   ├── gemini_engine.py              # Gemini triage chatbot + report writer
│   ├── config.py                     # All constants and configuration
│   ├── requirements.txt              # Python dependencies
│   ├── .env.example                  # Environment variable template
│   └── models/
│       └── neuropath_xception_best.h5    # ← place your model here
│
├── frontend/                         # React + Vite web application
│   ├── index.html
│   ├── vite.config.js                # Vite config + API proxy
│   ├── package.json
│   └── src/
│       ├── App.jsx                   # Root component + global state
│       ├── main.jsx                  # ReactDOM entry point
│       ├── api/
│       │   └── neuropath.js          # All fetch() calls to backend
│       ├── components/
│       │   ├── Sidebar.jsx           # Step nav + triage card + buttons
│       │   ├── ChatWindow.jsx        # Scrollable message thread
│       │   ├── MessageBubble.jsx     # User / AI message bubbles
│       │   ├── MRIResultCard.jsx     # Grad-CAM + heatmap + probabilities
│       │   ├── ReportPanel.jsx       # Full report + download button
│       │   └── InputBar.jsx          # Text input + MRI file upload
│       └── styles/
│           └── index.css             # Global dark clinical theme
│
└── README.md                         # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- A [Google Gemini API key](https://aistudio.google.com)
- The trained model file: `neuropath_xception_best.h5`

---

### 1. Clone the repository

```bash
git clone https://github.com/your-username/neuropath.git
cd neuropath
```

### 2. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Place your model file
mkdir models
# Copy neuropath_xception_best.h5 into backend/models/

# Set up your Gemini API key
# 1. Copy the .env.example file to create a .env file:
#    Windows
copy .env.example .env
#    Mac / Linux
cp .env.example .env

# 2. Open .env and add your Gemini API key:
#    GEMINI_API_KEY=your_key_here

# 3. Save the file. Do NOT commit .env to version control.

# Start the backend server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 --env-file .env
```

You should see:
```
✓  Model loaded from models/neuropath_xception_best.h5
✓  Grad-CAM engine ready
✓  Gemini triage model ready
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Frontend Setup

```bash
# Open a new terminal
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

You should see:
```
  VITE v5.x  ready

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://YOUR_IP:5173/
```

### 4. Open the app

- **Your machine:** http://localhost:5173
- **Same network:** http://YOUR_LOCAL_IP:5173

---

## 🖥️ How to Use

### Step 1 — Symptom Triage
1. Open the app and start typing your neurological symptoms in the chat
2. The AI will ask focused follow-up questions (2–4 questions max)
3. After 5–7 exchanges, a structured **Triage Summary** appears in the sidebar showing:
   - Risk Level (🔴 HIGH / 🟡 MEDIUM / 🟢 LOW)
   - Key symptoms, duration, severity
   - MRI recommendation

### Step 2 — MRI Analysis
1. Click the 📎 paperclip icon in the input bar
2. Upload a T1/T2-weighted axial brain MRI (JPG or PNG)
3. The result card appears inline in the chat showing:
   - Original MRI | Grad-CAM overlay | Activation heatmap
   - Prediction + confidence score
   - All class probabilities
   - Symptom-location correlation

### Step 3 — Full Report
1. Click **Generate Report** in the sidebar (enabled after MRI analysis)
2. Gemini writes a plain-language integrated report combining all findings
3. Click **Download** to save as a `.txt` file

---

## 📡 API Reference

The backend exposes a REST API at `http://localhost:8000`.
Interactive docs available at: **http://localhost:8000/docs**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Check server status |
| `POST` | `/api/chat` | Send a triage chat message |
| `POST` | `/api/analyse-mri` | Upload and analyse an MRI image |
| `POST` | `/api/report` | Generate full integrated report |
| `POST` | `/api/clear` | Reset the session |

### POST `/api/chat`

**Request:**
```json
{
  "message": "I've had severe headaches for 3 weeks",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

**Response:**
```json
{
  "reply": "Thank you for sharing that...",
  "triage_summary": "---TRIAGE SUMMARY---\nRISK LEVEL: MEDIUM\n...",
  "history": [...]
}
```

### POST `/api/analyse-mri`

**Request:** `multipart/form-data` with field `image` (JPG or PNG)

**Response:**
```json
{
  "pred_class": "glioma",
  "pred_label": "Glioma Tumour",
  "confidence": 94.3,
  "risk_flag": "HIGH PRIORITY — consult a neurologist urgently",
  "probabilities": { "Glioma Tumour": 0.943, ... },
  "gradcam_b64": "<base64 PNG>",
  "heatmap_b64": "<base64 PNG>",
  "correlation": "Symptom-Location Correlation for Glioma..."
}
```

---

## 🤝 Team

> Add your team members here

| Name | Role |
|---|---|
| — | ML Engineer — CNN training & Grad-CAM |
| — | AI Engineer — Gemini triage system |
| — | Backend Developer — FastAPI |
| — | Frontend Developer — React UI |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Brain Tumour MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — Kaggle
- [Xception Architecture](https://arxiv.org/abs/1610.02357) — François Chollet, Google
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391) — Selvaraju et al.
- [Google Gemini](https://deepmind.google/gemini) — Generative AI backbone
- [FastAPI](https://fastapi.tiangolo.com) — Modern Python web framework

---

<div align="center">

**Built with ❤️ for neurological AI research**

*NeuroPath AI — For screening purposes only. Must be reviewed by a licensed medical professional.*

</div>