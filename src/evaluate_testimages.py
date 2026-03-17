"""
evaluate_testimages.py
----------------------
Runs every trained model against all images in testimages/c0..c9 and writes
a CSV where:
  - each row = one image
  - columns = one prediction per model
  - last column = expected (true) class label (e.g. "c3")

Usage (from project root):
    python src/evaluate_testimages.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
TESTIMAGES_DIR = PROJECT_ROOT / "testimages"
OUTPUT_CSV = PROJECT_ROOT / "results" / "model_predictions.csv"

IMAGE_SIZE = (64, 64)
CLASS_NAMES = [f"c{i}" for i in range(10)]

# ── load models (same set as GUI) ─────────────────────────────────────────────

def load_models() -> dict:
    print("Loading models...")
    models = {
        "Logistic_Regression": joblib.load(MODELS_DIR / "logistic_model.pkl"),
        "Decision_Tree":       joblib.load(MODELS_DIR / "dt_model.pkl"),
        "Naive_Bayes":         joblib.load(MODELS_DIR / "nb_model.pkl"),
        "Random_Forest":       joblib.load(MODELS_DIR / "rf_model.pkl"),
        "SVM_PCA":             joblib.load(MODELS_DIR / "svm_model.pkl"),
        "_svm_scaler":         joblib.load(MODELS_DIR / "scaler.pkl"),
        "_svm_pca":            joblib.load(MODELS_DIR / "pca.pkl"),
        "CNN":                 tf.keras.models.load_model(MODELS_DIR / "cnn_model.keras"),
        "MobileNetV2_Transfer":tf.keras.models.load_model(MODELS_DIR / "transfer_model.keras"),
        "LSTM":                tf.keras.models.load_model(MODELS_DIR / "rnn_model.keras"),
        "Transformer":         tf.keras.models.load_model(MODELS_DIR / "transformer_model.keras"),
        "RL_Q_Network":        tf.keras.models.load_model(MODELS_DIR / "rl_q_model.keras"),
    }
    print("All models loaded.")
    return models


def is_normalized() -> bool:
    """Check whether processed training data is already 0-1 scaled."""
    sample = np.load(PROCESSED_DIR / "X_train.npy", mmap_mode="r")
    return float(np.max(sample[0])) <= 1.0


# ── image preprocessing (mirrors app.py) ─────────────────────────────────────

def preprocess(image_path: Path, normalize: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (flat_array, batch_array) ready for classical and deep models."""
    img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype="float32")
    if normalize:
        arr = arr / 255.0
    flat = arr.reshape(1, -1)          # (1, 64*64*3) for classical models
    batch = np.expand_dims(arr, 0)     # (1, 64, 64, 3) for Keras models
    return flat, batch


# ── per-model prediction (returns class string e.g. "c3") ────────────────────

def predict_all(flat: np.ndarray, batch: np.ndarray, models: dict) -> dict[str, str]:
    preds: dict[str, str] = {}

    for name in ("Logistic_Regression", "Decision_Tree", "Naive_Bayes", "Random_Forest"):
        idx = int(np.argmax(models[name].predict_proba(flat)[0]))
        preds[name] = CLASS_NAMES[idx]

    svm_feat = models["_svm_pca"].transform(models["_svm_scaler"].transform(flat))
    idx = int(np.argmax(models["SVM_PCA"].predict_proba(svm_feat)[0]))
    preds["SVM_PCA"] = CLASS_NAMES[idx]

    for name in ("CNN", "MobileNetV2_Transfer", "LSTM", "Transformer"):
        probs = models[name].predict(batch, verbose=0)[0]
        preds[name] = CLASS_NAMES[int(np.argmax(probs))]

    q_vals = models["RL_Q_Network"].predict(batch, verbose=0)[0]
    probs_rl = tf.nn.softmax(q_vals).numpy()
    preds["RL_Q_Network"] = CLASS_NAMES[int(np.argmax(probs_rl))]

    return preds


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    models = load_models()
    normalize = is_normalized()
    print(f"Image normalization: {'0-1 (÷255)' if normalize else 'raw 0-255'}")

    model_columns = [
        "Logistic_Regression",
        "Decision_Tree",
        "Naive_Bayes",
        "Random_Forest",
        "SVM_PCA",
        "CNN",
        "MobileNetV2_Transfer",
        "LSTM",
        "Transformer",
        "RL_Q_Network",
    ]

    rows: list[dict] = []
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for class_name in CLASS_NAMES:
        class_dir = TESTIMAGES_DIR / class_name
        if not class_dir.is_dir():
            print(f"  [skip] folder not found: {class_dir}")
            continue

        image_files = [
            p for p in sorted(class_dir.iterdir())
            if p.suffix.lower() in image_extensions
        ]
        print(f"  {class_name}: {len(image_files)} images")

        for img_path in image_files:
            flat, batch = preprocess(img_path, normalize)
            preds = predict_all(flat, batch, models)

            row = {"image": img_path.name}
            for col in model_columns:
                row[col] = preds[col]
            row["expected"] = class_name
            rows.append(row)

    fieldnames = ["image"] + model_columns + ["expected"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} images evaluated.")
    print(f"CSV saved to: {OUTPUT_CSV}")

    # ── quick accuracy summary per model ──────────────────────────────────────
    print("\n── Model accuracy on testimages ─────────────────────────────────")
    for col in model_columns:
        correct = sum(1 for r in rows if r[col] == r["expected"])
        print(f"  {col:<25} {correct}/{len(rows)}  ({correct/len(rows)*100:.1f}%)")


if __name__ == "__main__":
    main()
