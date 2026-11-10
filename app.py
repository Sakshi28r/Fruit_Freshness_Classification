"""
app.py  -  FruitFresh AI (Streamlit interface)

Run from the project root:
    streamlit run app.py

The app only LOADS the trained PNN (models/pnn_model.joblib); it never retrains.
"""

import streamlit as st

from src.predict import (BORDERLINE_CONFIDENCE, PredictionError, load_bundle,
                         predict_image)

st.set_page_config(page_title="FruitFresh AI", page_icon="🍎", layout="centered")

# ----------------------------------------------------------------------------
# light styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .result-card {padding: 1.1rem 1.3rem; border-radius: 14px; margin: 0.4rem 0 1rem 0;
                    border: 1px solid rgba(128,128,128,0.35);}
      .fresh  {background: rgba(46,160,67,0.15);  border-color: rgba(46,160,67,0.6);}
      .rotten {background: rgba(218,54,51,0.15);  border-color: rgba(218,54,51,0.6);}
      .result-title {font-size: 2rem; font-weight: 700; margin: 0;}
      .result-sub   {opacity: 0.8; margin: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def pct(p):
    """Format a 0-1 score as a percentage without showing a false 100.0% / 0.0%."""
    if p > 0.9995:
        return ">99.9%"
    if p < 0.0005:
        return "<0.1%"
    return f"{p:.1%}"


@st.cache_resource(show_spinner="Loading trained PNN model ...")
def get_bundle():
    """Loaded once and reused for every upload (no retraining)."""
    return load_bundle()


# ----------------------------------------------------------------------------
# header
# ----------------------------------------------------------------------------
st.title("🍎 FruitFresh AI")
st.caption("Fruit freshness classification with a Probabilistic Neural Network (PNN). "
           "Upload a photo of an apple, banana or strawberry.")

try:
    bundle = get_bundle()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

with st.sidebar:
    st.header("About the model")
    info = bundle["info"]
    st.write("**Classifier:** Probabilistic Neural Network, implemented from scratch in NumPy")
    st.write("**Layers:** input → pattern (Gaussian kernels) → summation → output (Bayes)")
    st.write(f"**Smoothing σ:** {info['freshness_sigma']:.3g}")
    st.write(f"**Class priors:** {info['freshness_priors']}")
    st.write(f"**Training images:** {info['n_train']}")
    st.write(f"**Features per image:** {info['n_features']}")
    st.info("Trained only on apples, bananas and strawberries. "
            "Other objects are outside what it has seen.")

# ----------------------------------------------------------------------------
# upload + predict
# ----------------------------------------------------------------------------
uploaded = st.file_uploader("Upload a fruit image",
                            type=["jpg", "jpeg", "png", "webp", "jfif"])

if uploaded is None:
    st.info("Choose an image to begin.")
    st.stop()

# Show a preview straight away (and catch files that are not real images)
try:
    from PIL import Image
    preview = Image.open(uploaded)
    preview.load()
    uploaded.seek(0)
    st.image(preview, caption="Uploaded image", width="stretch")
    uploaded.seek(0)
except Exception:
    st.error("This file could not be opened as an image. Please upload a valid JPG, PNG, WEBP or JFIF file.")
    st.stop()

if st.button("🔍 Predict", type="primary", width="stretch"):
    try:
        with st.spinner("Analysing ..."):
            result = predict_image(uploaded, bundle)
    except PredictionError as err:
        st.error(str(err))
        st.stop()

    css = "fresh" if result["prediction"] == "Fresh" else "rotten"
    icon = "✅" if result["prediction"] == "Fresh" else "⚠️"
    st.markdown(
        f"""
        <div class="result-card {css}">
          <p class="result-sub">Prediction</p>
          <p class="result-title">{icon} {result['prediction']}</p>
          <p class="result-sub">Fruit type (estimate): <b>{result['fruit_type']}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Model confidence", pct(result['confidence']))
    c2.metric("Fresh score", pct(result['p_fresh']))
    c3.metric("Rotten score", pct(result['p_rotten']))

    st.write("**Class posterior scores**")
    st.progress(result["p_fresh"], text=f"Fresh: {pct(result['p_fresh'])}")
    st.progress(result["p_rotten"], text=f"Rotten: {pct(result['p_rotten'])}")

    if result["borderline"]:
        st.warning(f"Borderline result: the winning score is below "
                   f"{BORDERLINE_CONFIDENCE:.0%}. The model is unsure about this image.")
    if result["unusual_image"]:
        st.warning("This image looks unlike the fruit photos the model was trained on "
                   "(different object, odd lighting or background). Treat the result with caution.")

    with st.expander("How to read the confidence score"):
        st.markdown(
            """
            The scores are the PNN's **normalised class posteriors**: each class's
            prior × Gaussian-kernel density, divided so the two scores add up to 100%.

            They are **model confidence, not calibrated probabilities**. A calibrated
            "90%" would mean about 9 in 10 such predictions are correct. PNN scores in
            many dimensions tend to be pushed towards 0% or 100%, so they can look more
            certain than the model really is. On the held-out test set the model's mean
            confidence was higher than its accuracy (see `outputs/metrics.json`).

            The fruit type is a separate PNN and is only an estimate.
            """
        )