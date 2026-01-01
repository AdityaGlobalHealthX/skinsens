from pathlib import Path
import os
import random

from flask import Flask, render_template, request, redirect, url_for
import numpy as np

# Prefer standalone Keras 3 loader (handles new-format H5)
try:
    import keras
    from keras.models import load_model  # type: ignore
except ImportError:
    from tensorflow.keras.models import load_model  # type: ignore
    import tensorflow as tf  # noqa: F401

from tensorflow.keras.preprocessing import image

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_densenet201.h5"
STATIC_IMG_DIR = BASE_DIR / "static" / "images"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
IMG_SIZE = (224, 224)

# Ensure directories exist
STATIC_IMG_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CLASS_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

LABEL_FULL = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}

# -----------------------------------------------------------------------------
# Load model once at startup
# -----------------------------------------------------------------------------
print("Loading model…")
MODEL = load_model(MODEL_PATH, compile=False)
print("Model loaded.")

# -----------------------------------------------------------------------------
# Flask application
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")


def preprocess_image(path: Path) -> np.ndarray:
    """Load and preprocess image for DenseNet."""
    img = image.load_img(path, target_size=IMG_SIZE)
    x = image.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probs = None
    filename = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = os.path.basename(file.filename)
            save_path = UPLOAD_DIR / filename
            file.save(save_path)

            x = preprocess_image(save_path)
            preds = MODEL.predict(x)[0]
            top_idx = int(np.argmax(preds))
            short_label = CLASS_LABELS[top_idx]
            prediction = LABEL_FULL[short_label]
            probs = None  # no longer used
        else:
            return redirect(url_for("index"))

    # choose a background image from static/images (copy once at startup)
    bg_images = list(STATIC_IMG_DIR.glob("*.jpg"))
    bg_url = None
    if bg_images:
        bg_url = url_for("static", filename=f"images/{random.choice(bg_images).name}")

    return render_template(
        "index.html",
        prediction=prediction,
        probs=probs,
        filename=filename,
        bg_url=bg_url,
    )


if __name__ == "__main__":
    app.run(debug=True)
