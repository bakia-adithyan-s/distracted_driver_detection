from __future__ import annotations

import sys
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
SRC_DIR = PROJECT_ROOT / "src"
STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
GENERATED_DIR = STATIC_DIR / "generated"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gradcam import generate_gradcam_output  # noqa: E402


app = Flask(__name__)
app.config["SECRET_KEY"] = "distracted-driver-frontend"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMAGE_SIZE = (64, 64)
CLASS_DETAILS = [
    ("c0", "safe driving"),
    ("c1", "texting - right"),
    ("c2", "talking on the phone - right"),
    ("c3", "texting - left"),
    ("c4", "talking on the phone - left"),
    ("c5", "operating the radio"),
    ("c6", "drinking"),
    ("c7", "reaching behind"),
    ("c8", "hair and makeup"),
    ("c9", "talking to passenger"),
]
LANDING_CARDS = [
    {
        "title": "Distraction Costs Lives",
        "text": "A distracted driver can miss hazards within seconds. Visual, manual, and cognitive distraction all raise crash risk.",
    },
    {
        "title": "Action Recognition Matters",
        "text": "This project identifies driver behavior from a single image and compares classical ML, deep learning, and RL-style models.",
    },
    {
        "title": "Model Comparison",
        "text": "Upload one driver image and inspect each model's predicted class together with probabilities for all 10 distraction classes.",
    },
    {
        "title": "Explainability Included",
        "text": "The result page also includes a CNN Grad-CAM visualization to show which image regions influenced the prediction.",
    },
]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@lru_cache(maxsize=1)
def input_is_normalized() -> bool:
    sample = np.load(PROCESSED_DIR / "X_train.npy", mmap_mode="r")
    return float(np.max(sample[0])) <= 1.0


@lru_cache(maxsize=1)
def load_models() -> dict[str, object]:
    return {
        "Logistic Regression": joblib.load(MODELS_DIR / "logistic_model.pkl"),
        "Decision Tree": joblib.load(MODELS_DIR / "dt_model.pkl"),
        "Naive Bayes": joblib.load(MODELS_DIR / "nb_model.pkl"),
        "Random Forest": joblib.load(MODELS_DIR / "rf_model.pkl"),
        "SVM (PCA)": joblib.load(MODELS_DIR / "svm_model.pkl"),
        "SVM Scaler": joblib.load(MODELS_DIR / "scaler.pkl"),
        "SVM PCA": joblib.load(MODELS_DIR / "pca.pkl"),
        "CNN": tf.keras.models.load_model(MODELS_DIR / "cnn_model.keras"),
        "MobileNetV2 Transfer": tf.keras.models.load_model(MODELS_DIR / "transfer_model.keras"),
        "LSTM": tf.keras.models.load_model(MODELS_DIR / "rnn_model.keras"),
        "Transformer": tf.keras.models.load_model(MODELS_DIR / "transformer_model.keras"),
        "RL Q-Network": tf.keras.models.load_model(MODELS_DIR / "rl_q_model.keras"),
    }


def class_label(index: int) -> str:
    code, meaning = CLASS_DETAILS[index]
    return f"{code}: {meaning}"


def preprocess_uploaded_image(file_storage) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(file_storage.stream).convert("RGB").resize(IMAGE_SIZE)
    raw_array = np.asarray(image, dtype="float32")
    model_array = raw_array / 255.0 if input_is_normalized() else raw_array.copy()
    return raw_array, model_array


def save_display_image(image_array: np.ndarray, target_path: Path) -> None:
    clipped = np.clip(image_array, 0, 255).astype("uint8")
    Image.fromarray(clipped).save(target_path)


def build_probability_rows(probabilities: np.ndarray) -> tuple[list[dict[str, object]], str]:
    predicted_index = int(np.argmax(probabilities))
    rows = []
    for index, (code, meaning) in enumerate(CLASS_DETAILS):
        rows.append(
            {
                "code": code,
                "meaning": meaning,
                "probability": float(probabilities[index]),
                "is_predicted": index == predicted_index,
            }
        )
    return rows, class_label(predicted_index)


def run_predictions(image_array: np.ndarray) -> list[dict[str, object]]:
    models = load_models()
    flat = image_array.reshape(1, -1)
    batch = np.expand_dims(image_array, axis=0)

    results: list[dict[str, object]] = []

    classical_specs = [
        ("Logistic Regression", models["Logistic Regression"], flat),
        ("Decision Tree", models["Decision Tree"], flat),
        ("Naive Bayes", models["Naive Bayes"], flat),
        ("Random Forest", models["Random Forest"], flat),
    ]
    for name, model, features in classical_specs:
        probabilities = model.predict_proba(features)[0]
        rows, predicted_label = build_probability_rows(probabilities)
        results.append(
            {
                "name": name,
                "family": "Classical ML",
                "rows": rows,
                "prediction": predicted_label,
            }
        )

    svm_features = models["SVM PCA"].transform(models["SVM Scaler"].transform(flat))
    svm_probabilities = models["SVM (PCA)"].predict_proba(svm_features)[0]
    rows, predicted_label = build_probability_rows(svm_probabilities)
    results.append(
        {
            "name": "SVM (PCA)",
            "family": "Classical ML",
            "rows": rows,
            "prediction": predicted_label,
        }
    )

    deep_specs = [
        ("CNN", models["CNN"]),
        ("MobileNetV2 Transfer", models["MobileNetV2 Transfer"]),
        ("LSTM", models["LSTM"]),
        ("Transformer", models["Transformer"]),
    ]
    for name, model in deep_specs:
        probabilities = model.predict(batch, verbose=0)[0]
        rows, predicted_label = build_probability_rows(probabilities)
        results.append(
            {
                "name": name,
                "family": "Deep Learning",
                "rows": rows,
                "prediction": predicted_label,
            }
        )

    rl_q_values = models["RL Q-Network"].predict(batch, verbose=0)[0]
    rl_probabilities = tf.nn.softmax(rl_q_values).numpy()
    rows, predicted_label = build_probability_rows(rl_probabilities)
    results.append(
        {
            "name": "RL Q-Network",
            "family": "Reinforcement Learning",
            "rows": rows,
            "prediction": predicted_label,
        }
    )

    return results


def get_majority_prediction(model_results: list[dict[str, object]]) -> dict[str, object] | None:
    votes: Counter[str] = Counter()
    class_meanings = {code: meaning for code, meaning in CLASS_DETAILS}
    class_order = {code: index for index, (code, _) in enumerate(CLASS_DETAILS)}

    for result in model_results:
        predicted_row = next((row for row in result["rows"] if row["is_predicted"]), None)
        if predicted_row is None:
            continue
        votes[str(predicted_row["code"])] += 1

    if not votes:
        return None

    winning_code, winning_votes = max(
        votes.items(),
        key=lambda item: (item[1], -class_order.get(item[0], 999)),
    )

    return {
        "code": winning_code,
        "meaning": class_meanings.get(winning_code, "unknown"),
        "votes": winning_votes,
        "total_models": len(model_results),
        "is_safe": winning_code == "c0",
    }


def generate_gradcam_asset(image_array: np.ndarray) -> str | None:
    try:
        cnn_model = load_models()["CNN"]
        gradcam_output = generate_gradcam_output(cnn_model, image_array, "conv2d_2")
        filename = f"gradcam_{uuid.uuid4().hex}.png"
        target = GENERATED_DIR / filename
        gradcam_uint8 = np.clip(gradcam_output * 255.0, 0, 255).astype("uint8")
        Image.fromarray(gradcam_uint8).save(target)
        return f"generated/{filename}"
    except Exception:
        return None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", cards=LANDING_CARDS, class_details=CLASS_DETAILS)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        flash("Please choose an image before submitting.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Supported file types are PNG, JPG, JPEG, and WEBP.")
        return redirect(url_for("index"))

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    upload_path = UPLOADS_DIR / unique_name

    raw_array, model_array = preprocess_uploaded_image(file)
    save_display_image(raw_array, upload_path)

    model_results = run_predictions(model_array)
    majority_prediction = get_majority_prediction(model_results)
    gradcam_path = generate_gradcam_asset(model_array)

    return render_template(
        "results.html",
        uploaded_image=url_for("static", filename=f"uploads/{unique_name}"),
        gradcam_image=url_for("static", filename=gradcam_path) if gradcam_path else None,
        model_results=model_results,
        majority_prediction=majority_prediction,
        class_details=CLASS_DETAILS,
    )


if __name__ == "__main__":
    app.run(debug=True)