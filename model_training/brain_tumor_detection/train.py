"""
NeuroPath — train.py
====================
Run this notebook cell-by-cell in Google Colab.

STEP 0  : Mount Drive + install deps
STEP 1  : Build data generators
STEP 2  : Build Xception model
STEP 3  : Phase 1 — train classifier head (base frozen)
STEP 4  : Phase 2 — fine-tune top layers
STEP 5  : Save model + plot learning curves
"""

# ─── STEP 0 : Setup ──────────────────────────────────────────────────────────
# Run this cell first in Colab

import os, random, warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax, Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                         ReduceLROnPlateau, CSVLogger)
from google.colab import drive

# Import project config
import sys
sys.path.append("/content/drive/MyDrive/NeuroPath")
from model_training.brain_tumor_detection.config import *

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# Mount Drive
drive.mount("/content/drive")

# Create output directories
for d in [MODEL_SAVE_DIR, PLOT_SAVE_DIR, GRADCAM_SAVE_DIR]:
    os.makedirs(d, exist_ok=True)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow version : {tf.__version__}")
print(f"GPU available      : {tf.config.list_physical_devices('GPU')}")
print(f"Classes            : {CLASS_NAMES}")


# ─── STEP 1 : Data Generators ────────────────────────────────────────────────

def build_generators():
    """
    Returns (train_gen, val_gen, test_gen) with class indices printed.
    """
    # Training + validation split from the Training folder
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,         # Extra robustness
        rotation_range=10,            # Small rotations (MRI can be slightly off-axis)
        zoom_range=0.1,
        validation_split=VAL_SPLIT,
    )

    # Test generator — NO augmentation, NO shuffle
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED,
    )

    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    print(f"\nClass indices : {train_gen.class_indices}")
    print(f"Train samples : {train_gen.samples}")
    print(f"Val   samples : {val_gen.samples}")
    print(f"Test  samples : {test_gen.samples}")

    return train_gen, val_gen, test_gen


train_gen, val_gen, test_gen = build_generators()


# ─── STEP 2 : Build Model ────────────────────────────────────────────────────

def build_model(trainable_base: bool = False) -> Model:
    """
    Xception backbone + custom classification head.

    Args:
        trainable_base: If False → only train the head (Phase 1).
                        If True  → entire network trains (Phase 2 fine-tuning).
    """
    base = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SHAPE,
        pooling=None,          # We add our own pooling below
    )
    base.trainable = trainable_base

    # Custom head — more expressive than the original for multi-class
    x = base.output
    x = GlobalAveragePooling2D()(x)     # Better than Flatten + MaxPool for CAM
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=outputs)
    return model, base


model, base_model = build_model(trainable_base=False)

model.compile(
    optimizer=Adamax(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)

model.summary()
print(f"\nTotal params     : {model.count_params():,}")
print(f"Trainable params : {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")


# ─── STEP 3 : Phase 1 — Train Head (Base Frozen) ─────────────────────────────

print("\n" + "="*60)
print("PHASE 1 — Training classifier head (Xception base frozen)")
print("="*60)

callbacks_phase1 = [
    ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
    CSVLogger(os.path.join(PLOT_SAVE_DIR, "phase1_log.csv")),
]

history1 = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks_phase1,
    verbose=1,
)


# ─── STEP 4 : Phase 2 — Fine-Tune Top Layers ─────────────────────────────────

print("\n" + "="*60)
print(f"PHASE 2 — Fine-tuning top {FINETUNE_LAYERS} layers of Xception")
print("="*60)

# Unfreeze the last N layers of the base
for layer in base_model.layers[-FINETUNE_LAYERS:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True  # Keep BN frozen to preserve learned statistics

# Recompile with a much smaller learning rate
model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)

print(f"Trainable params after unfreeze: "
      f"{sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

callbacks_phase2 = [
    ModelCheckpoint(
        filepath=BEST_MODEL_PATH,        # Overwrite only if val_accuracy improves
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-9,
        verbose=1,
    ),
    CSVLogger(os.path.join(PLOT_SAVE_DIR, "phase2_log.csv")),
]

history2 = model.fit(
    train_gen,
    epochs=FINETUNE_EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks_phase2,
    verbose=1,
)

# Save final model (last epoch weights, useful for continued training later)
model.save(FINAL_MODEL_PATH)
print(f"\nFinal model saved → {FINAL_MODEL_PATH}")
print(f"Best  model saved → {BEST_MODEL_PATH}")


# ─── STEP 5 : Plot Learning Curves ───────────────────────────────────────────

def merge_histories(h1, h2):
    """Concatenate Phase 1 and Phase 2 history dicts."""
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history[key]
    return merged


def plot_learning_curves(merged: dict, save_dir: str):
    metrics = [
        ("accuracy",  "val_accuracy",  "Accuracy",   "#4361EE"),
        ("loss",      "val_loss",      "Loss",        "#F72585"),
        ("auc",       "val_auc",       "AUC",         "#7209B7"),
        ("precision", "val_precision", "Precision",   "#3A0CA3"),
        ("recall",    "val_recall",    "Recall",      "#4CC9F0"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle("NeuroPath — Training History (Phase 1 + Phase 2)", fontsize=16, fontweight="bold")

    phase1_end = len(history1.history["accuracy"])  # Vertical separator

    for idx, (train_key, val_key, title, colour) in enumerate(metrics):
        ax = axes[idx]
        epochs_range = range(1, len(merged[train_key]) + 1)
        ax.plot(epochs_range, merged[train_key], label=f"Train {title}", color=colour, linewidth=2)
        ax.plot(epochs_range, merged[val_key],   label=f"Val {title}",   color=colour,
                linewidth=2, linestyle="--", alpha=0.7)
        ax.axvline(x=phase1_end, color="gray", linestyle=":", linewidth=1.5, label="Fine-tune start")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide the 6th subplot (we only have 5 metrics)
    axes[5].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved → {save_path}")


merged_history = merge_histories(history1, history2)
plot_learning_curves(merged_history, PLOT_SAVE_DIR)

# Quick final summary
print("\n" + "="*60)
print("TRAINING COMPLETE — Summary")
print("="*60)
best_val_acc = max(merged_history["val_accuracy"])
best_val_auc = max(merged_history["val_auc"])
print(f"Best Val Accuracy : {best_val_acc * 100:.2f}%")
print(f"Best Val AUC      : {best_val_auc:.4f}")
print(f"\nNext step → run evaluate.py to test on the held-out test set")
