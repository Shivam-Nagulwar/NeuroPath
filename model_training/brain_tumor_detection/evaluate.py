"""
NeuroPath — evaluate.py
=======================
Run this AFTER train.py is complete and you're satisfied with the training curves.

What this script produces
─────────────────────────
1. Test-set accuracy, loss, AUC, Precision, Recall
2. Full classification report (per-class F1, precision, recall)
3. Confusion matrix heatmap (saved as PNG)
4. Grad-CAM heatmap visualisation on sample test images
5. Per-class Grad-CAM gallery (4 samples per class)

All outputs are saved to DRIVE_BASE/plots/ and DRIVE_BASE/gradcam_outputs/.
"""

import os, sys, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix

from google.colab import drive
drive.mount("/content/drive")

sys.path.append("/content/drive/MyDrive/NeuroPath")
from model_training.brain_tumor_detection.config import *

tf.get_logger().setLevel("ERROR")


# ─── 1. Load Model + Test Generator ─────────────────────────────────────────

print("Loading best model …")
model = load_model(BEST_MODEL_PATH)
print(f"Model loaded from : {BEST_MODEL_PATH}")

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# Map folder index → class name (folder order may differ from CLASS_NAMES)
idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
print(f"Class index map : {idx_to_class}")


# ─── 2. Evaluate on Test Set ─────────────────────────────────────────────────

print("\nEvaluating on test set …")
results = model.evaluate(test_gen, verbose=1)
metric_names = model.metrics_names

print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)
for name, val in zip(metric_names, results):
    print(f"  {name:<12} : {val:.4f}")


# ─── 3. Classification Report ────────────────────────────────────────────────

print("\nGenerating predictions …")
test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
target_names = [idx_to_class[i] for i in sorted(idx_to_class)]
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))


# ─── 4. Confusion Matrix ─────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    cm_matrix = confusion_matrix(y_true, y_pred)
    cm_percent = cm_matrix.astype("float") / cm_matrix.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("NeuroPath — Confusion Matrix (Test Set)", fontsize=15, fontweight="bold")

    # Raw counts
    sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0],
                linewidths=0.5, cbar=True)
    axes[0].set_title("Raw Counts", fontsize=13)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Normalised percentages
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1],
                linewidths=0.5, cbar=True)
    axes[1].set_title("Normalised (%)", fontsize=13)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix saved → {path}")


plot_confusion_matrix(y_true, y_pred,
                      class_names=target_names,
                      save_dir=PLOT_SAVE_DIR)


# ─── 5. Grad-CAM Engine ──────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for Keras models.

    How it works
    ─────────────
    1. Forward pass: run the image through the model, capturing the output of
       the target convolutional layer (GRADCAM_LAYER_NAME from config).
    2. Backward pass: compute the gradient of the predicted class score
       with respect to every feature map in that layer.
    3. Pool the gradients spatially (global average) → importance weight per map.
    4. Weighted sum of feature maps → raw heatmap.
    5. ReLU + normalise → final heatmap in [0, 1].
    6. Resize to input image size and overlay as a colour heatmap.
    """

    def __init__(self, model: tf.keras.Model, layer_name: str = GRADCAM_LAYER_NAME):
        self.model = model
        self.layer_name = layer_name
        # Build a sub-model that outputs both the conv layer AND the final prediction
        self.grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output],
        )

    def compute(self, img_array: np.ndarray, class_idx: int = None) -> np.ndarray:
        """
        Args:
            img_array : Preprocessed image, shape (1, H, W, 3), values in [0, 1].
            class_idx : Target class index. If None, uses the predicted class.

        Returns:
            heatmap : np.ndarray of shape (H, W) with values in [0, 1].
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_score = predictions[:, class_idx]

        # Gradients of the class score w.r.t. the conv layer output
        grads = tape.gradient(class_score, conv_outputs)

        # Global average pool the gradients → weight per feature map
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight each feature map by its importance
        conv_outputs = conv_outputs[0]  # Shape: (h, w, C)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU + normalise to [0, 1]
        heatmap = tf.nn.relu(heatmap).numpy()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    def overlay(self,
                original_img: np.ndarray,
                heatmap: np.ndarray,
                alpha: float = GRADCAM_ALPHA) -> np.ndarray:
        """
        Resize heatmap to match original image and blend as a colour overlay.

        Args:
            original_img : uint8 RGB image, shape (H, W, 3).
            heatmap      : Float array from self.compute(), shape (h, w).
            alpha        : Blend strength (0 = original only, 1 = heatmap only).

        Returns:
            overlaid : uint8 RGB image with heatmap burned in.
        """
        H, W = original_img.shape[:2]

        # Resize heatmap to original image dimensions
        heatmap_resized = cv2.resize(heatmap, (W, H))

        # Apply JET colormap (blue=cold/low, red=hot/high activation)
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]       # Drop alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8) # Scale to [0, 255]

        # Weighted blend
        overlaid = cv2.addWeighted(original_img, 1 - alpha,
                                   heatmap_colored, alpha, 0)
        return overlaid


# ─── 6. Grad-CAM on Sample Test Images ───────────────────────────────────────

def load_and_preprocess(img_path: str) -> tuple:
    """Returns (preprocessed_array, original_uint8_array)."""
    img = load_img(img_path, target_size=IMG_SIZE)
    original = np.array(img)                             # uint8, for display
    preprocessed = img_to_array(img) / 255.0             # float [0,1]
    preprocessed = np.expand_dims(preprocessed, axis=0) # batch dim
    return preprocessed, original


def gradcam_gallery(model, test_dir: str, save_dir: str,
                    samples_per_class: int = 4):
    """
    For each class, pick N sample images, run Grad-CAM, and save a gallery figure.
    """
    gradcam = GradCAM(model)
    os.makedirs(save_dir, exist_ok=True)

    for class_name in CLASS_NAMES:
        class_folder = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_folder):
            print(f"  Warning: folder not found — {class_folder}")
            continue

        images = [f for f in os.listdir(class_folder)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(images) == 0:
            continue

        selected = random.sample(images, min(samples_per_class, len(images)))

        fig, axes = plt.subplots(len(selected), 3,
                                 figsize=(12, 4 * len(selected)))
        if len(selected) == 1:
            axes = [axes]

        fig.suptitle(
            f"Grad-CAM Gallery — {CLASS_LABELS.get(class_name, class_name)}",
            fontsize=14, fontweight="bold"
        )

        for row, img_name in enumerate(selected):
            img_path = os.path.join(class_folder, img_name)
            preprocessed, original = load_and_preprocess(img_path)

            # Predict
            preds = model.predict(preprocessed, verbose=0)[0]
            pred_idx = np.argmax(preds)
            pred_class = idx_to_class[pred_idx]
            confidence = preds[pred_idx] * 100

            # Grad-CAM
            heatmap = gradcam.compute(preprocessed, class_idx=pred_idx)
            overlay = gradcam.overlay(original, heatmap)

            ax_orig, ax_heat, ax_over = axes[row]

            ax_orig.imshow(original)
            ax_orig.set_title("Original MRI", fontsize=11)
            ax_orig.axis("off")

            ax_heat.imshow(heatmap, cmap="jet")
            ax_heat.set_title("Grad-CAM Heatmap", fontsize=11)
            ax_heat.axis("off")

            ax_over.imshow(overlay)
            correct = "✓" if pred_class == class_name else "✗"
            ax_over.set_title(
                f"Overlay | Pred: {pred_class} {correct}\n"
                f"Confidence: {confidence:.1f}%",
                fontsize=10,
                color="green" if pred_class == class_name else "red"
            )
            ax_over.axis("off")

            # Add a colourbar for the heatmap column
            cbar = fig.colorbar(cm.ScalarMappable(cmap="jet"),
                                ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.set_label("Activation", fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"gradcam_{class_name}.png")
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.show()
        print(f"  Saved → {save_path}")


print("\nGenerating Grad-CAM galleries for all classes …")
gradcam_gallery(model, TEST_DIR, GRADCAM_SAVE_DIR, samples_per_class=4)


# ─── 7. Single-Image Grad-CAM (Interactive) ──────────────────────────────────

def analyse_single_image(img_path: str):
    """
    Run the full NeuroPath pipeline on a single MRI image and display results.
    Call this interactively: analyse_single_image('/path/to/mri.jpg')
    """
    gradcam = GradCAM(model)
    preprocessed, original = load_and_preprocess(img_path)

    preds = model.predict(preprocessed, verbose=0)[0]
    pred_idx   = np.argmax(preds)
    pred_class = idx_to_class[pred_idx]
    confidence = preds[pred_idx] * 100

    heatmap = gradcam.compute(preprocessed, class_idx=pred_idx)
    overlay = gradcam.overlay(original, heatmap)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"NeuroPath Analysis  |  {CLASS_LABELS.get(pred_class, pred_class)}  "
        f"|  Confidence: {confidence:.1f}%",
        fontsize=14, fontweight="bold"
    )

    axes[0].imshow(original);   axes[0].set_title("Input MRI");          axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM (what the AI sees)"); axes[1].axis("off")
    axes[2].imshow(overlay);    axes[2].set_title("Overlay on MRI");      axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    # ── Confidence Bar Chart ──
    fig2, ax = plt.subplots(figsize=(8, 4))
    colours = ["#4CC9F0" if i != pred_idx else "#F72585"
               for i in range(NUM_CLASSES)]
    bars = ax.barh(target_names, preds * 100, color=colours)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Class Probability Distribution")
    ax.set_xlim(0, 105)
    for bar, val in zip(bars, preds * 100):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)
    plt.tight_layout()
    plt.show()

    # ── Text Report ──
    print("\n" + "="*55)
    print("NEUROPATH DIAGNOSTIC REPORT")
    print("="*55)
    print(f"Prediction  : {CLASS_LABELS.get(pred_class, pred_class)}")
    print(f"Confidence  : {confidence:.2f}%")
    print(f"\nClass probabilities:")
    for cls, prob in zip(target_names, preds):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<15} {bar:<32} {prob*100:5.2f}%")
    print(f"\nDescription:\n  {CLASS_DESCRIPTIONS.get(pred_class, '')}")
    print("="*55)
    print("DISCLAIMER: This is an AI screening tool only.")
    print("Always consult a qualified radiologist / neurologist.")
    print("="*55)

    return pred_class, confidence, preds


# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
print("Outputs saved to:")
print(f"  Plots       : {PLOT_SAVE_DIR}")
print(f"  Grad-CAM    : {GRADCAM_SAVE_DIR}")
print(f"\nTo analyse a custom MRI:")
print("  analyse_single_image('/path/to/your/mri.jpg')")
print("\nWhen results are satisfactory → next step is app.py (Gradio UI)")
