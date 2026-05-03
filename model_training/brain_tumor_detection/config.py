"""
NeuroPath — config.py
=====================
Single source of truth for all paths, hyperparameters, and constants.
Edit ONLY this file when you want to change any setting.
"""

import os

# ─── Google Drive Paths ──────────────────────────────────────────────────────
DRIVE_BASE   = "/content/drive/MyDrive/NeuroPath"
DATASET_DIR  = "/content/drive/MyDrive/BrainTumer_Detection/brain-tumor-mri-dataset"
TRAIN_DIR    = os.path.join(DATASET_DIR, "Training")
TEST_DIR     = os.path.join(DATASET_DIR, "Testing")

# Where model checkpoints and outputs are saved
MODEL_SAVE_DIR  = os.path.join(DRIVE_BASE, "saved_models")
PLOT_SAVE_DIR   = os.path.join(DRIVE_BASE, "plots")
GRADCAM_SAVE_DIR = os.path.join(DRIVE_BASE, "gradcam_outputs")

# Final model file path
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "neuropath_xception_best.h5")
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "neuropath_xception_final.h5")

# ─── Dataset ─────────────────────────────────────────────────────────────────
# Must match the exact folder names inside Training/ and Testing/
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# ─── Model & Training Hyperparameters ────────────────────────────────────────
IMG_SIZE    = (299, 299)       # Xception native input size
IMG_SHAPE   = (299, 299, 3)
BATCH_SIZE  = 32
EPOCHS      = 20               # Increase to 30 for best results
LEARNING_RATE = 0.001
VAL_SPLIT   = 0.2              # 20 % of Training folder used for validation

# ─── Fine-Tuning Phase (Phase 2) ─────────────────────────────────────────────
# After initial training, unfreeze top N layers of Xception for fine-tuning
FINETUNE_LAYERS   = 30         # Unfreeze last 30 layers
FINETUNE_LR       = 1e-5       # Much smaller LR to avoid destroying weights
FINETUNE_EPOCHS   = 10

# ─── Grad-CAM ────────────────────────────────────────────────────────────────
# Name of the last convolutional block in Xception — used as the Grad-CAM target
GRADCAM_LAYER_NAME = "block14_sepconv2_act"
GRADCAM_ALPHA      = 0.5       # Heatmap blend strength (0 = no overlay, 1 = full)

# ─── Display Labels (for UI and reports) ─────────────────────────────────────
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

# Brain lobe → symptom correlation map (used in Module 4)
LOBE_SYMPTOM_MAP = {
    "frontal":    ["personality changes", "impaired judgment", "difficulty speaking", "weakness"],
    "parietal":   ["numbness", "difficulty reading/writing", "spatial disorientation"],
    "temporal":   ["memory problems", "hearing changes", "language difficulties", "seizures"],
    "occipital":  ["vision loss", "visual hallucinations", "difficulty recognising objects"],
    "cerebellum": ["balance problems", "coordination issues", "tremors", "dizziness"],
    "brainstem":  ["swallowing difficulty", "facial weakness", "double vision", "vomiting"],
    "pituitary":  ["hormonal imbalance", "vision changes", "fatigue", "headaches"],
}

# ─── Kaggle Setup (Required for Dataset Download) ─────────────────────────────
# Before running this script in Colab:
# 1. Go to https://www.kaggle.com/account
# 2. Create API token (downloads kaggle.json)
# 3. Upload kaggle.json to Colab or run:
#    from google.colab import files
#    files.upload()  # Upload kaggle.json
# 4. Move it: !mkdir -p ~/.kaggle && !mv kaggle.json ~/.kaggle/ && !chmod 600 ~/.kaggle/kaggle.json
