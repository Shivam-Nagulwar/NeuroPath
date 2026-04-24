"""
NeuroPath — config.py
=====================
Adapted from Google Colab version for local FastAPI backend.
Edit ONLY this file when you want to change any setting.
"""

import os

# ─── Local Paths ─────────────────────────────────────────────────────────────
# Put your .h5 model file inside the backend/models/ folder
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR  = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH       = os.path.join(MODEL_SAVE_DIR, "neuropath_xception_best.h5")

# ─── Gatekeeper Model (brain MRI validator) ───────────────────────────────────
# MobileNetV2 binary classifier: brain_mri (0) vs non_brain_mri (1)
# Place gatekeeper_model.h5 inside backend/models/
GATEKEEPER_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "gatekeeper_model.h5")
GATEKEEPER_IMG_SIZE   = (224, 224)          # MobileNetV2 input size
GATEKEEPER_THRESHOLD  = 0.5                 # sigmoid > 0.5 = non_brain_mri → reject

# ─── Dataset ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# ─── Model Input ─────────────────────────────────────────────────────────────
IMG_SIZE   = (299, 299)        # Xception native input size
IMG_SHAPE  = (299, 299, 3)

# ─── Grad-CAM ────────────────────────────────────────────────────────────────
GRADCAM_LAYER_NAME = "block14_sepconv2_act"
GRADCAM_ALPHA      = 0.5

# ─── Display Labels ──────────────────────────────────────────────────────────
CLASS_LABELS = {
    "glioma":     "Glioma Tumour",
    "meningioma": "Meningioma Tumour",
    "notumor":    "No Tumour Detected",
    "pituitary":  "Pituitary Tumour",
}

CLASS_DESCRIPTIONS = {
    "glioma": (
        "Glioma originates in the glial (supportive) cells of the brain or spine. "
        "It accounts for roughly 30% of all brain tumours and ~80% of malignant ones. "
        "Common symptoms include headaches, seizures, and cognitive changes. "
        "Requires urgent neurological consultation."
    ),
    "meningioma": (
        "Meningioma grows from the meninges — the protective membranes surrounding the brain. "
        "Most are benign and slow-growing. Symptoms depend on location and include "
        "vision problems, headaches, and weakness. Often monitored or surgically removed."
    ),
    "notumor": (
        "No tumour was detected in this MRI scan. The model found no abnormal mass "
        "or lesion in the provided image. However, this is a screening tool — "
        "always consult a qualified radiologist for a definitive diagnosis."
    ),
    "pituitary": (
        "Pituitary tumours (adenomas) form in the pituitary gland at the brain's base. "
        "They can disrupt hormone regulation, causing vision issues, fatigue, or "
        "hormonal imbalances. Most are benign and highly treatable."
    ),
}

# ─── Lobe → Symptom Correlation ──────────────────────────────────────────────
LOBE_SYMPTOM_MAP = {
    "frontal":    ["personality changes", "impaired judgment", "difficulty speaking", "weakness"],
    "parietal":   ["numbness", "difficulty reading/writing", "spatial disorientation"],
    "temporal":   ["memory problems", "hearing changes", "language difficulties", "seizures"],
    "occipital":  ["vision loss", "visual hallucinations", "difficulty recognising objects"],
    "cerebellum": ["balance problems", "coordination issues", "tremors", "dizziness"],
    "brainstem":  ["swallowing difficulty", "facial weakness", "double vision", "vomiting"],
    "pituitary":  ["hormonal imbalance", "vision changes", "fatigue", "headaches"],
}

# ─── Tumour → Lobe Map ───────────────────────────────────────────────────────
TUMOUR_LOBE_MAP = {
    "glioma":     ["frontal", "temporal", "parietal"],
    "meningioma": ["frontal", "parietal", "cerebellum"],
    "pituitary":  ["pituitary"],
    "notumor":    [],
}