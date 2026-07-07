import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path("outputs/final_model.keras")
CLASS_NAMES_PATH = Path("outputs/class_names.json")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def prepare_image(image, image_size):
    image = image.convert("RGB")
    image = image.resize(image_size)
    image_array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(image_array, axis=0)


def format_label(label):
    return label.replace("_", " ").title()


def probability_rows(class_names, predictions):
    rows = [
        {
            "Class": format_label(class_name),
            "Probability": float(probability),
            "Confidence": f"{float(probability) * 100:.2f}%",
        }
        for class_name, probability in zip(class_names, predictions)
    ]
    return sorted(rows, key=lambda row: row["Probability"], reverse=True)


def render_probability_bars(rows):
    bars = []
    for row in rows:
        percent = row["Probability"] * 100
        bars.append(
            '<div class="probability-row">'
            '<div class="probability-label">'
            f'<span>{row["Class"]}</span>'
            f"<strong>{percent:.2f}%</strong>"
            "</div>"
            '<div class="probability-track">'
            f'<div class="probability-fill" style="width: {percent:.2f}%;"></div>'
            "</div>"
            "</div>"
        )

    st.markdown(
        f'<div class="probability-list">{"".join(bars)}</div>',
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Food Classifier",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --app-bg: #f7f8f5;
            --panel-bg: #ffffff;
            --ink: #1c221f;
            --muted: #66736d;
            --line: #dfe5dd;
            --accent: #0f7b63;
            --accent-soft: #e7f4ef;
            --accent-strong: #085f4b;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 12%, rgba(15, 123, 99, 0.10), transparent 28rem),
                linear-gradient(180deg, #fbfcf8 0%, var(--app-bg) 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.4rem;
            padding: 0.2rem 0 1.4rem;
        }

        .eyebrow {
            color: var(--accent-strong);
            font-size: 0.78rem;
            font-weight: 760;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
        }

        .hero h1 {
            color: var(--ink);
            font-size: clamp(2.1rem, 4.3vw, 4.2rem);
            font-weight: 820;
            letter-spacing: 0;
            line-height: 1.02;
            margin: 0;
        }

        .hero p {
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.65;
            max-width: 720px;
            margin: 0.85rem 0 0;
        }

        .result-panel {
            background: #10231e;
            border-radius: 8px;
            color: #ffffff;
            padding: 1.3rem;
        }

        .result-panel .label {
            color: rgba(255, 255, 255, 0.72);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .result-panel h2 {
            color: #ffffff;
            font-size: clamp(1.7rem, 3vw, 2.6rem);
            letter-spacing: 0;
            margin: 0.25rem 0 0.4rem;
        }

        .result-panel p {
            color: rgba(255, 255, 255, 0.72);
            margin: 0;
        }

        .info-strip {
            display: grid;
            gap: 0.8rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin: 1rem 0 1.2rem;
        }

        .info-item {
            background: var(--accent-soft);
            border: 1px solid #cfe9df;
            border-radius: 8px;
            padding: 0.85rem;
        }

        .info-item strong {
            color: var(--ink);
            display: block;
            font-size: 0.98rem;
            line-height: 1.25;
        }

        .info-item span {
            color: var(--muted);
            display: block;
            font-size: 0.84rem;
            margin-top: 0.25rem;
        }

        .probability-list {
            display: grid;
            gap: 0.85rem;
            margin: 1rem 0;
        }

        .probability-label {
            align-items: center;
            display: flex;
            gap: 1rem;
            justify-content: space-between;
            margin-bottom: 0.35rem;
        }

        .probability-label span {
            color: var(--ink);
            font-size: 0.94rem;
            font-weight: 690;
        }

        .probability-label strong {
            color: var(--accent-strong);
            font-size: 0.88rem;
            font-weight: 800;
        }

        .probability-track {
            background: #e5ebe7;
            border-radius: 999px;
            height: 0.62rem;
            overflow: hidden;
        }

        .probability-fill {
            background: linear-gradient(90deg, #0f7b63, #45a46f);
            border-radius: 999px;
            height: 100%;
        }

        .stButton > button {
            background: var(--accent);
            border: 1px solid var(--accent);
            border-radius: 8px;
            color: #ffffff;
            font-weight: 760;
            min-height: 3rem;
            width: 100%;
        }

        .stButton > button:hover {
            background: var(--accent-strong);
            border-color: var(--accent-strong);
            color: #ffffff;
        }

        [data-testid="stFileUploader"] {
            background: var(--panel-bg);
            border: 1px dashed #b9c7c0;
            border-radius: 8px;
            padding: 0.8rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--accent-strong);
            font-weight: 800;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .info-strip {
                grid-template-columns: 1fr;
            }

            .result-panel {
                padding: 1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">CNN Image Recognition</div>
        <h1>Food Image Classifier</h1>
        <p>
            Upload a dish photo and the trained model will identify the most likely
            food type.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    st.error("Model files were not found.")
    st.info("Train the model first by running `python train.py`, then reopen this app.")
    st.stop()

model = load_model()
class_names = load_class_names()
input_height = model.input_shape[1]
input_width = model.input_shape[2]

st.markdown(
    f"""
    <div class="info-strip">
        <div class="info-item">
            <strong>{len(class_names)} classes</strong>
            <span>Available trained labels</span>
        </div>
        <div class="info-item">
            <strong>{input_width} x {input_height}</strong>
            <span>Model input resolution</span>
        </div>
        <div class="info-item">
            <strong>Local model</strong>
            <span>TensorFlow/Keras prediction</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_column, result_column = st.columns([1.05, 0.95], gap="large")

with upload_column:
    st.subheader("Image")
    uploaded_file = st.file_uploader(
        "Upload JPG, JPEG, or PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_container_width=True)
    else:
        st.info("Upload a photo to begin.")

    predict_clicked = st.button("Run prediction", disabled=image is None)

with result_column:
    st.subheader("Prediction")

    if image is None:
        st.empty()
        st.write("Results will appear here after you upload an image.")
    elif predict_clicked:
        with st.spinner("Analyzing image..."):
            input_image = prepare_image(image, (input_width, input_height))
            predictions = model.predict(input_image, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])
        predicted_label = format_label(class_names[predicted_index])
        rows = probability_rows(class_names, predictions)

        st.markdown(
            f"""
            <div class="result-panel">
                <div class="label">Top prediction</div>
                <h2>{predicted_label}</h2>
                <p>{confidence * 100:.2f}% model confidence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_left, metric_right = st.columns(2)
        metric_left.metric("Confidence", f"{confidence * 100:.2f}%")
        metric_right.metric("Alternatives", f"{max(len(class_names) - 1, 0)}")

        render_probability_bars(rows)

        st.dataframe(
            [{"Class": row["Class"], "Confidence": row["Confidence"]} for row in rows],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.write("The uploaded image is ready. Run prediction to classify it.")
