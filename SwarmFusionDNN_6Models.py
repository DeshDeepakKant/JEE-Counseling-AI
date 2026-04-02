"""
SwarmFusionDNN — Extended to 6 Base Models
============================================
Original paper: "Swarm intelligence optimization-based fusion of ConvMixer-enhanced
deep neural networks for brain tumor detection" (Asif et al.)

Extension: Adds InceptionV3 and ResNet50 as two additional base models beyond the
original four (MobileNet, DenseNet121, MobileNetV2, Xception).

Pipeline:
  1. Data loading & augmentation (BR35H / Figshare / Bangladesh datasets)
  2. Build 6 ConvMixer-enhanced CNN models
  3. Train each model independently
  4. Run PSO to find optimal ensemble weights (minimize classifier error rate)
  5. Final weighted-average ensemble prediction
  6. Evaluation (Accuracy, Sensitivity, Specificity, Precision, F1)
  7. Grad-CAM visualisation

Usage:
  python SwarmFusionDNN_6Models.py --dataset BR35H --data_dir /path/to/dataset
"""

# ─── 0. IMPORTS ─────────────────────────────────────────────────────────────
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.applications import (
    MobileNet,
    DenseNet121,
    MobileNetV2,
    Xception,
    InceptionV3,   # ← NEW model 5
    ResNet50,       # ← NEW model 6
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score,
)

print(f"TensorFlow version : {tf.__version__}")
print(f"GPUs available     : {tf.config.list_physical_devices('GPU')}")

# ─── 1. CONFIGURATION ────────────────────────────────────────────────────────
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    DATA_DIR    = "/home/anya/.cache/kagglehub/datasets/ahmedhamada0/brain-tumor-detection/versions/12"   # override via --data_dir
    OUTPUT_DIR  = "outputs"
    WEIGHTS_DIR = "model_weights"

    # ── Image settings ───────────────────────────────────────────────────
    IMG_SIZE    = 224          # input resolution for all models
    # InceptionV3 / ResNet50 both support 224×224
    BATCH_SIZE  = 32           # reduced from 64 to fit most 8-GB GPUs

    # ── Training ─────────────────────────────────────────────────────────
    EPOCHS      = 20
    LR          = 1e-3
    PATIENCE    = 12
    ALPHA_DROP  = 0.5

    # ── PSO hyper-parameters ─────────────────────────────────────────────
    PSO_POP     = 50
    PSO_ITER    = 100
    PSO_W       = 0.5          # inertia weight
    PSO_C1      = 1.0          # cognitive coefficient
    PSO_C2      = 2.0          # social coefficient
    PSO_WMIN    = 0.0          # weight lower bound
    PSO_WMAX    = 1.0          # weight upper bound

    # ── Models ───────────────────────────────────────────────────────────
    MODEL_NAMES = [
        "MobileNet",
        "DenseNet121",
        "MobileNetV2",
        "Xception",
        "InceptionV3",   # new
        "ResNet50",       # new
    ]
    N_MODELS    = len(MODEL_NAMES)

    # ── Dataset ───────────────────────────────────────────────────────────
    DATASET     = "BR35H"      # BR35H | Figshare | Bangladesh
    N_CLASSES   = 2            # set to 3 for multi-class datasets
    CLASS_NAMES = ["Normal", "Tumor"]


def make_dirs(cfg: Config):
    for d in [cfg.OUTPUT_DIR, cfg.WEIGHTS_DIR]:
        os.makedirs(d, exist_ok=True)


# ─── 2. DATA PIPELINE ────────────────────────────────────────────────────────

def build_generators(cfg: Config):
    """
    Build Keras ImageDataGenerators for train and test splits.
    Applies augmentation consistent with the paper:
      - Horizontal/vertical flips
      - Shear (0.2)
      - Rotation (±20°)
    """
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,
        rotation_range=20,
        validation_split=0.2,   # 80/20 split implicit in generator
    )
    test_aug = ImageDataGenerator(rescale=1.0 / 255.0)

    target = (cfg.IMG_SIZE, cfg.IMG_SIZE)

    train_gen = train_aug.flow_from_directory(
        cfg.DATA_DIR,
        target_size=target,
        batch_size=cfg.BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )
    val_gen = train_aug.flow_from_directory(
        cfg.DATA_DIR,
        target_size=target,
        batch_size=cfg.BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )
    # For final test predictions (same path, no augmentation, no split)
    test_gen = test_aug.flow_from_directory(
        cfg.DATA_DIR,
        target_size=target,
        batch_size=cfg.BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    cfg.N_CLASSES   = train_gen.num_classes
    cfg.CLASS_NAMES = list(train_gen.class_indices.keys())
    print(f"Classes detected: {cfg.CLASS_NAMES}")
    return train_gen, val_gen, test_gen


# ─── 3. MODEL BUILDING ───────────────────────────────────────────────────────

def _convmixer_block(x, filters: int):
    """
    ConvMixer block appended to the base CNN feature extractor.

    Architecture:
        Depthwise Conv (kernel=3, padding=same)  → captures spatial correlations
        Pointwise Conv (1×1)                      → shapes channel interactions
        GeLU activation                           → non-linearity
        BatchNormalization                        → training stability
    """
    # Depthwise convolution — spatial mixing
    x = layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.Conv2D(filters, kernel_size=1, use_bias=False)(x)   # Pointwise
    x = layers.Activation("gelu")(x)
    x = layers.BatchNormalization()(x)
    return x


def build_model(base_fn, cfg: Config, model_name: str) -> Model:
    """
    Build one ConvMixer-enhanced model.

    Steps
    -----
    1. Load pre-trained backbone (ImageNet, no top, input 224×224×3).
    2. Freeze backbone weights.
    3. Append a ConvMixer block.
    4. GlobalAveragePooling → Alpha Dropout → Dense classification head.
    """
    inp = layers.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3), name="input")

    # ── Backbone ─────────────────────────────────────────────────────────────
    # InceptionV3 expects a minimum of 75×75 so 224 is fine.
    backbone = base_fn(
        include_top=False,
        weights="imagenet",
        input_tensor=inp,
    )
    backbone.trainable = False   # freeze for transfer learning

    x = backbone.output

    # ── ConvMixer block ───────────────────────────────────────────────────────
    filters = x.shape[-1]        # keep channel dimension from backbone
    x = _convmixer_block(x, filters)

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.AlphaDropout(cfg.ALPHA_DROP)(x)
    out = layers.Dense(cfg.N_CLASSES, activation="softmax", name="predictions")(x)

    model = Model(inputs=inp, outputs=out, name=model_name)
    model.compile(
        optimizer=Adam(learning_rate=cfg.LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Map model name → Keras application function
# InceptionV3 and ResNet50 are the two new additions.
MODEL_REGISTRY = {
    "MobileNet"   : MobileNet,
    "DenseNet121" : DenseNet121,
    "MobileNetV2" : MobileNetV2,
    "Xception"    : Xception,
    "InceptionV3" : InceptionV3,   # NEW
    "ResNet50"    : ResNet50,       # NEW
}


def build_all_models(cfg: Config):
    models = {}
    for name in cfg.MODEL_NAMES:
        print(f"  Building {name} …")
        models[name] = build_model(MODEL_REGISTRY[name], cfg, name)
    return models


# ─── 4. TRAINING ─────────────────────────────────────────────────────────────

def _callbacks(cfg: Config, model_name: str):
    ckpt_path = os.path.join(cfg.WEIGHTS_DIR, f"{model_name}_best.keras")
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_all_models(models: dict, train_gen, val_gen, cfg: Config):
    histories = {}
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"  Training {name}")
        print(f"{'='*60}")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=cfg.EPOCHS,
            callbacks=_callbacks(cfg, name),
            verbose=1,
        )
        histories[name] = history
        # Evaluate immediately
        loss, acc = model.evaluate(val_gen, verbose=0)
        print(f"  {name} → Val Accuracy: {acc*100:.2f}%  |  Val Loss: {loss:.4f}")
    return histories


def load_or_train(models: dict, train_gen, val_gen, cfg: Config):
    """
    Load saved weights if they exist, otherwise train from scratch.
    """
    for name, model in models.items():
        ckpt = os.path.join(cfg.WEIGHTS_DIR, f"{name}_best.keras")
        if os.path.exists(ckpt):
            print(f"  Loading saved weights for {name} from {ckpt}")
            model.load_weights(ckpt)
        else:
            print(f"  No checkpoint found for {name}. Training from scratch.")
    # Train only models without checkpoints
    missing = [
        n for n in models
        if not os.path.exists(os.path.join(cfg.WEIGHTS_DIR, f"{n}_best.keras"))
    ]
    if missing:
        print(f"\n  Models to train: {missing}")
        histories = {}
        for name in missing:
            print(f"\n{'='*60}\n  Training {name}\n{'='*60}")
            h = models[name].fit(
                train_gen,
                validation_data=val_gen,
                epochs=cfg.EPOCHS,
                callbacks=_callbacks(cfg, name),
                verbose=1,
            )
            histories[name] = h
        return histories
    print("\n  All model weights loaded from disk. Skipping training.")
    return {}


# ─── 5. PREDICTION ───────────────────────────────────────────────────────────

def get_predictions(models: dict, generator, cfg: Config) -> np.ndarray:
    """
    Collect raw softmax probability arrays from all 6 models.
    Returns array of shape (N_MODELS, n_samples, n_classes).
    """
    generator.reset()
    all_preds = []
    for name in cfg.MODEL_NAMES:
        print(f"  Predicting with {name} …")
        generator.reset()
        preds = models[name].predict(generator, verbose=0)
        all_preds.append(preds)
    return np.array(all_preds)   # (6, n_samples, n_classes)


def get_true_labels(generator) -> np.ndarray:
    generator.reset()
    return generator.classes


# ─── 6. PSO OPTIMISATION ─────────────────────────────────────────────────────

class PSO:
    """
    Particle Swarm Optimisation to find optimal ensemble weights
    that minimise the Classifier Error Rate across 6 models.

    Particle dimension = N_MODELS (one weight per model).
    Constraint: each weight ∈ [0, 1].

    Objective (minimise):
        error_rate = (misclassified / total) × 100
    where the ensemble prediction is:
        Y_pred = argmax( Σ w_i · p_i )
    """

    def __init__(self, cfg: Config):
        self.n_models  = cfg.N_MODELS
        self.pop       = cfg.PSO_POP
        self.max_iter  = cfg.PSO_ITER
        self.w         = cfg.PSO_W
        self.c1        = cfg.PSO_C1
        self.c2        = cfg.PSO_C2
        self.wmin      = cfg.PSO_WMIN
        self.wmax      = cfg.PSO_WMAX

    def _ensemble_pred(self, weights: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """Weighted average: probs shape (n_models, n_samples, n_classes)."""
        w = weights / (weights.sum() + 1e-9)          # normalise to sum to 1
        weighted = np.einsum("m,mnc->nc", w, probs)   # (n_samples, n_classes)
        return np.argmax(weighted, axis=1)

    def _error_rate(self, weights: np.ndarray, probs: np.ndarray,
                    true_labels: np.ndarray) -> float:
        preds = self._ensemble_pred(weights, probs)
        return (np.sum(preds != true_labels) / len(true_labels)) * 100.0

    def optimise(self, probs: np.ndarray, true_labels: np.ndarray):
        """
        Run PSO optimisation.

        Parameters
        ----------
        probs       : (N_MODELS, n_samples, n_classes) – softmax outputs
        true_labels : (n_samples,) – ground-truth class indices

        Returns
        -------
        best_weights : (N_MODELS,) – optimal per-model weights
        best_error   : float       – final minimum error rate (%)
        error_history: list[float] – error per iteration (for plotting)
        """
        dim    = self.n_models
        # ── Initialise particles ──────────────────────────────────────────
        pos  = np.random.uniform(self.wmin, self.wmax, (self.pop, dim))
        vel  = np.zeros_like(pos)
        pbest_pos  = pos.copy()
        pbest_err  = np.array([self._error_rate(p, probs, true_labels) for p in pos])
        gbest_idx  = np.argmin(pbest_err)
        gbest_pos  = pbest_pos[gbest_idx].copy()
        gbest_err  = pbest_err[gbest_idx]

        error_history = [gbest_err]
        print(f"\n  PSO Start — Initial best error rate: {gbest_err:.2f}%")

        for iteration in range(self.max_iter):
            r1 = np.random.rand(self.pop, dim)
            r2 = np.random.rand(self.pop, dim)

            # Velocity update (standard PSO formula)
            vel = (
                self.w * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (gbest_pos - pos)
            )
            pos = np.clip(pos + vel, self.wmin, self.wmax)

            # Evaluate fitness
            errors = np.array([self._error_rate(p, probs, true_labels) for p in pos])

            # Update personal bests
            improved = errors < pbest_err
            pbest_pos[improved] = pos[improved]
            pbest_err[improved] = errors[improved]

            # Update global best
            new_best_idx = np.argmin(pbest_err)
            if pbest_err[new_best_idx] < gbest_err:
                gbest_pos = pbest_pos[new_best_idx].copy()
                gbest_err = pbest_err[new_best_idx]

            error_history.append(gbest_err)

            if (iteration + 1) % 10 == 0:
                print(f"  Iter {iteration+1:>3}/{self.max_iter} — "
                      f"Best Error: {gbest_err:.4f}%  "
                      f"Best Accuracy: {100 - gbest_err:.4f}%")

        print(f"\n  PSO finished — Optimal error rate: {gbest_err:.4f}%")
        print(f"  Optimal accuracy          : {100 - gbest_err:.4f}%")
        # Normalise weights so they sum to 1 for readability
        norm_weights = gbest_pos / gbest_pos.sum()
        print("\n  Optimal weights (normalised):")
        for n, w in zip(Config.MODEL_NAMES, norm_weights):
            print(f"    {n:<15}: {w:.4f}")

        return gbest_pos, gbest_err, error_history


# ─── 7. EVALUATION ───────────────────────────────────────────────────────────

def ensemble_predict(weights: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Weighted-average prediction using PSO-derived weights."""
    w = weights / (weights.sum() + 1e-9)
    weighted = np.einsum("m,mnc->nc", w, probs)
    return np.argmax(weighted, axis=1)


def evaluate(true_labels: np.ndarray, pred_labels: np.ndarray,
             probs: np.ndarray, weights: np.ndarray,
             class_names: list, cfg: Config):
    """
    Compute and display comprehensive metrics:
      Accuracy, Sensitivity (Recall), Specificity, Precision, F1, AUC.
    """
    print("\n" + "="*60)
    print("  SWARMFUSIONDNN (6 Models) — EVALUATION RESULTS")
    print("="*60)

    acc = accuracy_score(true_labels, pred_labels) * 100
    f1  = f1_score(true_labels, pred_labels, average="macro") * 100

    avg_type = "binary" if cfg.N_CLASSES == 2 else "macro"
    prec = precision_score(true_labels, pred_labels, average=avg_type, zero_division=0) * 100
    rec  = recall_score(true_labels, pred_labels, average=avg_type, zero_division=0) * 100

    # Specificity from confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    if cfg.N_CLASSES == 2:
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) * 100
    else:
        # Per-class specificity → macro average
        spec_list = []
        for i in range(cfg.N_CLASSES):
            tn_i = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp_i = cm[:, i].sum() - cm[i, i]
            spec_list.append(tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0)
        spec = np.mean(spec_list) * 100

    # AUC
    w_norm = weights / weights.sum()
    ensemble_probs = np.einsum("m,mnc->nc", w_norm, probs)
    try:
        if cfg.N_CLASSES == 2:
            auc = roc_auc_score(true_labels, ensemble_probs[:, 1])
        else:
            auc = roc_auc_score(
                true_labels, ensemble_probs, multi_class="ovr", average="macro"
            )
    except Exception:
        auc = float("nan")

    metrics = {
        "Accuracy (%)":    acc,
        "Sensitivity (%)": rec,
        "Specificity (%)": spec,
        "Precision (%)":   prec,
        "F1-Score (%)":    f1,
        "AUC":             auc,
        "Error Rate (%)":  100 - acc,
    }

    for k, v in metrics.items():
        print(f"  {k:<25}: {v:.4f}")

    print("\n  Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    # ── Individual model performance ──────────────────────────────────────
    print("\n  Individual model accuracies (before PSO fusion):")
    for i, name in enumerate(cfg.MODEL_NAMES):
        m_pred = np.argmax(probs[i], axis=1)
        m_acc  = accuracy_score(true_labels, m_pred) * 100
        print(f"    {name:<15}: {m_acc:.2f}%")

    return metrics


# ─── 8. VISUALISATION ────────────────────────────────────────────────────────

def plot_training_histories(histories: dict, cfg: Config):
    if not histories:
        print("No training histories to plot (weights loaded from disk).")
        return
    n = len(histories)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]
    for ax_row, (name, hist) in zip(axes, histories.items()):
        ax_row[0].plot(hist.history["accuracy"], label="Train")
        ax_row[0].plot(hist.history["val_accuracy"], label="Val")
        ax_row[0].set_title(f"{name} — Accuracy")
        ax_row[0].legend(); ax_row[0].grid(True)

        ax_row[1].plot(hist.history["loss"], label="Train")
        ax_row[1].plot(hist.history["val_loss"], label="Val")
        ax_row[1].set_title(f"{name} — Loss")
        ax_row[1].legend(); ax_row[1].grid(True)

    plt.tight_layout()
    path = os.path.join(cfg.OUTPUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    print(f"  Training curves saved → {path}")
    plt.show()


def plot_pso_convergence(error_history: list, cfg: Config):
    plt.figure(figsize=(9, 4))
    plt.plot(error_history, color="royalblue", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Error Rate (%)")
    plt.title("PSO Convergence — Classifier Error Rate")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(cfg.OUTPUT_DIR, "pso_convergence.png")
    plt.savefig(path, dpi=150)
    print(f"  PSO convergence plot saved → {path}")
    plt.show()


def plot_confusion_matrix(true_labels, pred_labels, class_names, cfg: Config):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("SwarmFusionDNN (6 Models) — Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(cfg.OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    print(f"  Confusion matrix saved → {path}")
    plt.show()


def plot_model_comparison(probs: np.ndarray, true_labels: np.ndarray,
                          weights: np.ndarray, cfg: Config):
    """Bar chart comparing individual model accuracy vs. ensemble accuracy."""
    names  = cfg.MODEL_NAMES + ["SwarmFusionDNN\n(PSO Ensemble)"]
    accs   = []
    for i in range(cfg.N_MODELS):
        m_pred = np.argmax(probs[i], axis=1)
        accs.append(accuracy_score(true_labels, m_pred) * 100)
    # Ensemble
    ens_pred = ensemble_predict(weights, probs)
    accs.append(accuracy_score(true_labels, ens_pred) * 100)

    colors = ["#4C72B0"] * cfg.N_MODELS + ["#DD8452"]
    plt.figure(figsize=(11, 5))
    bars = plt.bar(names, accs, color=colors, edgecolor="white", width=0.6)
    plt.ylabel("Accuracy (%)")
    plt.ylim(max(0, min(accs) - 5), 101)
    plt.title("Individual Models vs. SwarmFusionDNN Ensemble (6 Models)")
    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontsize=9,
        )
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(cfg.OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"  Model comparison chart saved → {path}")
    plt.show()


def plot_pso_weights(weights: np.ndarray, cfg: Config):
    """Pie / bar of final PSO weights per model."""
    norm_w = weights / weights.sum()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(cfg.MODEL_NAMES, norm_w, color=plt.cm.tab10.colors[:cfg.N_MODELS],
                   edgecolor="white")
    plt.ylabel("Normalised PSO Weight")
    plt.title("PSO-Optimised Weights per Base Model (6 Models)")
    for bar, wv in zip(bars, norm_w):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{wv:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    plt.ylim(0, max(norm_w) * 1.2)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(cfg.OUTPUT_DIR, "pso_weights.png")
    plt.savefig(path, dpi=150)
    print(f"  PSO weights chart saved → {path}")
    plt.show()


# ─── 9. GRAD-CAM ─────────────────────────────────────────────────────────────

def make_gradcam_heatmap(img_array: np.ndarray, model: Model,
                         last_conv_name: str, pred_index: int = None) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a single image.

    Parameters
    ----------
    img_array     : (1, H, W, 3) preprocessed image
    model         : trained Keras model
    last_conv_name: name of the last convolutional layer in the model
    pred_index    : class index to visualise (defaults to top prediction)
    """
    grad_model = Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads    = tape.gradient(class_channel, conv_outputs)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap  = conv_outputs[0] @ pooled[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap)
    heatmap  = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy()


def _last_conv_name(model: Model) -> str:
    """Find the last Conv2D or DepthwiseConv2D layer by name."""
    for layer in reversed(model.layers):
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No convolutional layer found in model.")


def show_gradcam(model: Model, img_path: str, class_names: list,
                 model_name: str, output_dir: str):
    """Load an image, run Grad-CAM, and save the overlay."""
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_arr = keras.preprocessing.image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, 0)

    last_conv = _last_conv_name(model)
    heatmap   = make_gradcam_heatmap(img_arr, model, last_conv)

    # Overlay
    heatmap_resized = np.uint8(255 * heatmap)
    jet   = cm.get_cmap("jet")
    jet_c = jet(np.arange(256))[:, :3]
    jet_heatmap = keras.preprocessing.image.array_to_img(
        jet_c[heatmap_resized.reshape(-1)].reshape(heatmap.shape + (3,)) * 255
    )
    jet_heatmap = jet_heatmap.resize((224, 224))
    jet_arr     = keras.preprocessing.image.img_to_array(jet_heatmap)

    orig_arr    = keras.preprocessing.image.img_to_array(img)
    overlay     = jet_arr * 0.4 + orig_arr * 0.6
    overlay_img = keras.preprocessing.image.array_to_img(overlay)

    pred = model.predict(img_arr, verbose=0)
    label = class_names[np.argmax(pred)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img);         axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(overlay_img); axes[1].set_title(f"Grad-CAM ({label})"); axes[1].axis("off")
    plt.suptitle(f"Grad-CAM — {model_name}")
    plt.tight_layout()
    fname = os.path.join(output_dir, f"gradcam_{model_name}.png")
    plt.savefig(fname, dpi=150)
    print(f"  Grad-CAM saved → {fname}")
    plt.show()


# ─── 10. MAIN PIPELINE ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SwarmFusionDNN — 6 Model Ensemble")
    p.add_argument("--dataset",  default="BR35H",
                   choices=["BR35H", "Figshare", "Bangladesh"],
                   help="Dataset to use")
    p.add_argument("--data_dir", default=None,
                   help="Root directory of dataset (class subdirs inside)")
    p.add_argument("--epochs",   type=int, default=20)
    p.add_argument("--batch",    type=int, default=32)
    p.add_argument("--gradcam_img", default=None,
                   help="Path to a single image for Grad-CAM visualisation")
    return p.parse_args()


def run(cfg: Config):
    make_dirs(cfg)

    print("\n" + "█"*60)
    print("  SwarmFusionDNN — 6 Model Ensemble")
    print(f"  Dataset : {cfg.DATASET}")
    print(f"  Models  : {cfg.MODEL_NAMES}")
    print("█"*60 + "\n")

    # ── Data ──────────────────────────────────────────────────────────────
    print("[ Step 1 ] Building data generators …")
    train_gen, val_gen, test_gen = build_generators(cfg)

    # ── Models ────────────────────────────────────────────────────────────
    print("\n[ Step 2 ] Building 6 ConvMixer-enhanced models …")
    models = build_all_models(cfg)
    print(f"\n  Total models: {len(models)}")
    for name, m in models.items():
        total = m.count_params()
        print(f"    {name:<15}: {total:,} parameters")

    # ── Training ──────────────────────────────────────────────────────────
    print("\n[ Step 3 ] Training / loading models …")
    histories = load_or_train(models, train_gen, val_gen, cfg)
    plot_training_histories(histories, cfg)

    # ── Inference ─────────────────────────────────────────────────────────
    print("\n[ Step 4 ] Generating predictions from all 6 models …")
    val_gen.reset()
    probs       = get_predictions(models, val_gen, cfg)
    true_labels = get_true_labels(val_gen)
    print(f"  Probability tensor shape: {probs.shape}   (N_models, N_samples, N_classes)")

    # ── PSO ───────────────────────────────────────────────────────────────
    print("\n[ Step 5 ] Running PSO to find optimal ensemble weights …")
    pso = PSO(cfg)
    best_weights, best_error, error_history = pso.optimise(probs, true_labels)
    plot_pso_convergence(error_history, cfg)
    plot_pso_weights(best_weights, cfg)

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n[ Step 6 ] Evaluating ensemble …")
    pred_labels = ensemble_predict(best_weights, probs)
    metrics = evaluate(true_labels, pred_labels, probs, best_weights, cfg.CLASS_NAMES, cfg)
    plot_confusion_matrix(true_labels, pred_labels, cfg.CLASS_NAMES, cfg)
    plot_model_comparison(probs, true_labels, best_weights, cfg)

    # ── Save results ──────────────────────────────────────────────────────
    results_path = os.path.join(cfg.OUTPUT_DIR, "results.csv")
    pd.DataFrame([metrics]).to_csv(results_path, index=False)
    print(f"\n  Results saved → {results_path}")

    weights_path = os.path.join(cfg.OUTPUT_DIR, "pso_weights.npy")
    np.save(weights_path, best_weights)
    print(f"  PSO weights saved → {weights_path}")

    print("\n[ Step 7 ] Pipeline complete! ✓")
    return models, best_weights, probs, true_labels, metrics


if __name__ == "__main__":
    args = parse_args()

    cfg = Config()
    cfg.DATASET   = args.dataset
    if args.data_dir:
        cfg.DATA_DIR = args.data_dir
    cfg.EPOCHS    = args.epochs
    cfg.BATCH_SIZE = args.batch

    # Dataset-specific class counts
    if cfg.DATASET == "BR35H":
        cfg.N_CLASSES = 2
    else:
        cfg.N_CLASSES = 3

    models, best_weights, probs, true_labels, metrics = run(cfg)

    # Optional Grad-CAM on a single image
    if args.gradcam_img:
        print("\n[ Grad-CAM ] Running on:", args.gradcam_img)
        for name, model in models.items():
            show_gradcam(
                model, args.gradcam_img, cfg.CLASS_NAMES,
                model_name=name, output_dir=cfg.OUTPUT_DIR,
            )
