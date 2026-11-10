"""
predict.py
----------
Loads the saved PNN bundle (models/pnn_model.joblib) and classifies ONE image.
Nothing is retrained here - the stored training patterns are simply reused.

Command-line use (from the project root):
    python -m src.predict path\\to\\fruit.jpg
"""

import sys
from pathlib import Path

import joblib
import numpy as np

from src.feature_extraction import extract_features
from src.pnn import _squared_distances
from src.preprocessing import IMAGE_EXTENSIONS, PROJECT_ROOT, load_image

MODEL_PATH = PROJECT_ROOT / "models" / "pnn_model.joblib"

# Below this winning posterior the app calls the result "borderline".
BORDERLINE_CONFIDENCE = 0.70


class PredictionError(Exception):
    """Raised with a user-friendly message when an image cannot be processed."""


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_bundle(path=MODEL_PATH):
    """Load the trained model + scaler. Run `python -m src.train` first."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. "
            "Run  python -m src.train  first.")
    bundle = joblib.load(path)
    bundle["novelty_threshold"] = _novelty_threshold(bundle)
    return bundle


def _nearest_train_distance(model, X):
    """Distance from each row of X to its closest stored training pattern."""
    all_patterns = np.vstack([model.patterns_[c] for c in model.classes_])
    return np.sqrt(_squared_distances(X, all_patterns).min(axis=1))


def _novelty_threshold(bundle):
    """
    'Does this image look like the training images at all?'
    For every training pattern we find its nearest OTHER training pattern; the
    99th percentile of those distances (x1.25) is the 'normal' limit. An upload
    farther than that from everything we trained on (e.g. a photo of a car)
    gets a warning, because a PNN will still output a confident-looking
    posterior for any input.
    """
    model = bundle["freshness_model"]
    P = np.vstack([model.patterns_[c] for c in model.classes_])
    d2 = _squared_distances(P, P)
    np.fill_diagonal(d2, np.inf)                       # ignore self-distance
    nearest = np.sqrt(d2.min(axis=1))
    return float(np.percentile(nearest, 99) * 1.25)


# ----------------------------------------------------------------------------
# prediction
# ----------------------------------------------------------------------------
def predict_image(source, bundle):
    """
    source : file path or an uploaded file object
    Returns a dictionary with everything the app needs to display.
    """
    # Reject unsupported files by name when we have one
    name = getattr(source, "name", source if isinstance(source, (str, Path)) else "")
    suffix = Path(str(name)).suffix.lower()
    if suffix and suffix not in IMAGE_EXTENSIONS:
        raise PredictionError(
            f"Unsupported file type '{suffix}'. Please upload a JPG, PNG, WEBP or JFIF image.")

    try:
        img = load_image(source, size=tuple(bundle["image_size"]))
    except Exception:
        raise PredictionError("This file could not be read as an image. "
                              "It may be corrupted or not a real picture.")

    x = bundle["scaler"].transform(extract_features(img)[None, :])

    # Freshness PNN (layers 1-4 run inside predict_proba)
    proba = bundle["freshness_model"].predict_proba(x)[0]
    classes = bundle["class_names"]                    # ['Fresh', 'Rotten']
    best = int(np.argmax(proba))

    # Fruit-type PNN
    fruit_proba = bundle["fruit_model"].predict_proba(x)[0]
    fruit_names = bundle["fruit_names"]
    fruit_best = int(np.argmax(fruit_proba))

    distance = float(_nearest_train_distance(bundle["freshness_model"], x)[0])

    return {
        "prediction": classes[best],
        "confidence": float(proba[best]),
        "p_fresh": float(proba[classes.index("Fresh")]),
        "p_rotten": float(proba[classes.index("Rotten")]),
        "fruit_type": fruit_names[fruit_best],
        "fruit_confidence": float(fruit_proba[fruit_best]),
        "borderline": bool(proba[best] < BORDERLINE_CONFIDENCE),
        "unusual_image": bool(distance > bundle["novelty_threshold"]),
        "nearest_train_distance": distance,
        "image": img,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <image_path>")
        sys.exit(1)
    try:
        out = predict_image(sys.argv[1], load_bundle())
    except (PredictionError, FileNotFoundError) as e:
        print("Error:", e)
        sys.exit(1)
    print(f"Fruit type (estimate): {out['fruit_type']}  ({out['fruit_confidence']:.1%})")
    print(f"Prediction           : {out['prediction']}")
    print(f"P(Fresh)  = {out['p_fresh']:.4f}")
    print(f"P(Rotten) = {out['p_rotten']:.4f}")
    print(f"Model confidence     : {out['confidence']:.1%}  (normalised PNN posterior, not calibrated)")
    if out["borderline"]:
        print("Note: borderline result.")
    if out["unusual_image"]:
        print("Warning: this image looks unlike the training images.")