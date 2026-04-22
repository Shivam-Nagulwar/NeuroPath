"""
NeuroPath — cnn_engine.py
=========================
CNN inference pipeline + Grad-CAM engine.
Identical logic to the Colab version, stripped of Gradio dependencies.
"""

import base64
import io

import cv2
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    CLASS_LABELS,
    CLASS_DESCRIPTIONS,
    GRADCAM_ALPHA,
    GRADCAM_LAYER_NAME,
    IMG_SIZE,
    LOBE_SYMPTOM_MAP,
    NUM_CLASSES,
    TUMOUR_LOBE_MAP,
)

tf.get_logger().setLevel("ERROR")

# ─── Class index map (sorted to match Keras ImageDataGenerator order) ─────────
CLASS_IDX_MAP = {i: name for i, name in enumerate(sorted(CLASS_NAMES))}


# ===============================================================================
# Grad-CAM Engine
# ===============================================================================

class GradCAM:
    def __init__(self, model, layer_name: str = GRADCAM_LAYER_NAME):
        self.model = model
        self.grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output],
        )

    def compute(self, img_array: np.ndarray, class_idx: int = None) -> np.ndarray:
        with tf.GradientTape() as tape:
            conv_out, preds = self.grad_model(img_array)
            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]))
            score = preds[:, class_idx]
        grads  = tape.gradient(score, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.nn.relu(heatmap).numpy()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap

    def overlay(self, original_rgb: np.ndarray, heatmap: np.ndarray,
                alpha: float = GRADCAM_ALPHA) -> np.ndarray:
        H, W    = original_rgb.shape[:2]
        resized = cv2.resize(heatmap, (W, H))
        colored = (cm.jet(resized)[:, :, :3] * 255).astype(np.uint8)
        return cv2.addWeighted(original_rgb, 1 - alpha, colored, alpha, 0)


# ===============================================================================
# Model + Engine — loaded once at import time
# ===============================================================================

print("Loading NeuroPath CNN model …")
cnn_model = load_model(BEST_MODEL_PATH)
print(f"  ✓  Model loaded from {BEST_MODEL_PATH}")
print(f"  ✓  Class map: {CLASS_IDX_MAP}")

gradcam_engine = GradCAM(cnn_model)
print("  ✓  Grad-CAM engine ready")


# ===============================================================================
# Helper — PIL Image → base64 PNG string
# ===============================================================================

def _pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ===============================================================================
# Symptom-Location Correlation
# ===============================================================================

def correlate_symptoms(symptom_text: str, pred_class: str) -> str:
    if pred_class == "notumor" or not symptom_text.strip():
        return "No tumour detected — no location correlation applicable."

    symptom_lower = symptom_text.lower()
    lobes   = TUMOUR_LOBE_MAP.get(pred_class, [])
    matches = []

    for lobe in lobes:
        triggered = [s for s in LOBE_SYMPTOM_MAP.get(lobe, []) if s in symptom_lower]
        if triggered:
            matches.append(
                f"{lobe.capitalize()} lobe → "
                f"Your reported symptoms ({', '.join(triggered)}) "
                f"are consistent with this location."
            )

    if matches:
        return (
            f"Symptom-Location Correlation for {CLASS_LABELS[pred_class]}:\n\n"
            + "\n".join(matches)
            + "\n\nIndicative only — a neurologist must confirm via MRI report."
        )

    return (
        f"No direct symptom-location match found for {CLASS_LABELS[pred_class]}. "
        f"This tumour type commonly affects the "
        f"{', '.join(l.capitalize() for l in lobes)} region(s). "
        f"Symptoms vary widely depending on tumour size and growth direction."
    )


# ===============================================================================
# Main CNN Inference — returns plain Python dict (JSON-serialisable)
# ===============================================================================

def run_cnn_analysis(pil_image: Image.Image, symptom_text: str = "") -> dict:
    """
    Runs CNN prediction + Grad-CAM on a PIL image.

    Returns
    -------
    dict with keys:
        pred_class      str   e.g. "glioma"
        pred_label      str   e.g. "Glioma Tumour"
        confidence      float 0-100
        description     str
        risk_flag       str
        probabilities   dict  { display_label: float 0-1 }
        gradcam_b64     str   base64-encoded PNG
        heatmap_b64     str   base64-encoded PNG
        correlation     str
    """
    # Pre-process
    img        = pil_image.convert("RGB").resize(IMG_SIZE)
    arr        = img_to_array(img) / 255.0
    preprocessed = np.expand_dims(arr, axis=0)
    original_rgb = np.array(img)

    # Predict
    preds      = cnn_model.predict(preprocessed, verbose=0)[0]
    pred_idx   = int(np.argmax(preds))
    pred_class = CLASS_IDX_MAP[pred_idx]
    confidence = float(preds[pred_idx]) * 100

    # Grad-CAM
    heatmap     = gradcam_engine.compute(preprocessed, class_idx=pred_idx)
    overlay_arr = gradcam_engine.overlay(original_rgb, heatmap)
    gradcam_pil = Image.fromarray(overlay_arr)
    heatmap_pil = Image.fromarray(
        (cm.jet(cv2.resize(heatmap, IMG_SIZE))[:, :, :3] * 255).astype(np.uint8)
    )

    # Probabilities dict  (display label → float)
    all_probs = {
        CLASS_LABELS[CLASS_IDX_MAP[i]]: float(preds[i])
        for i in range(NUM_CLASSES)
    }

    # Risk flag
    if pred_class == "notumor":
        risk_flag = "No tumour detected"
    elif pred_class == "glioma":
        risk_flag = "HIGH PRIORITY — consult a neurologist urgently"
    else:
        risk_flag = "ABNORMALITY DETECTED — schedule a neurology consultation"

    # Symptom correlation
    correlation = correlate_symptoms(symptom_text, pred_class)

    return {
        "pred_class":    pred_class,
        "pred_label":    CLASS_LABELS[pred_class],
        "confidence":    round(confidence, 2),
        "description":   CLASS_DESCRIPTIONS[pred_class],
        "risk_flag":     risk_flag,
        "probabilities": all_probs,
        "gradcam_b64":   _pil_to_b64(gradcam_pil),
        "heatmap_b64":   _pil_to_b64(heatmap_pil),
        "correlation":   correlation,
    }
