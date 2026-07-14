import base64
import hashlib
import json
import math
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from database import (
    clear_predictions,
    fetch_predictions,
    init_database,
    save_prediction,
    save_uploaded_image,
)


MODEL_PATH = Path("outputs/final_model.keras")
CLASS_NAMES_PATH = Path("outputs/class_names.json")
CLASS_IMAGE_ROOT = Path("food-5/test")
PREFERRED_CLASS_ORDER = ["fried_rice", "sushi", "ice_cream", "hamburger", "cheesecake"]
PAGE_SIZE = 6


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def image_to_data_uri(path):
    path = Path(path)
    if not path.exists():
        return ""

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{path.suffix.lower().lstrip('.')};base64,{encoded}"


def prepare_image(image, image_size):
    image = image.convert("RGB")
    image = image.resize(image_size)
    image_array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(image_array, axis=0)


def format_label(label):
    return label.replace("_", " ").title()


def class_display_order(class_names):
    preferred = [class_name for class_name in PREFERRED_CLASS_ORDER if class_name in class_names]
    remaining = [class_name for class_name in class_names if class_name not in preferred]
    return preferred + remaining


def representative_image(class_name):
    class_dir = CLASS_IMAGE_ROOT / class_name
    if not class_dir.exists():
        return None

    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        image_paths = sorted(class_dir.glob(pattern))
        if image_paths:
            return image_paths[0]

    return None


def probability_rows(class_names, predictions):
    rows = [
        {
            "Class": format_label(class_name),
            "Raw Class": class_name,
            "Probability": float(probability),
            "Confidence": f"{float(probability) * 100:.2f}%",
        }
        for class_name, probability in zip(class_names, predictions)
    ]
    return sorted(rows, key=lambda row: row["Probability"], reverse=True)


def confidence_tier(probability):
    if probability >= 0.8:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


def prediction_reason(rows):
    top = rows[0]
    second = rows[1] if len(rows) > 1 else None
    top_percent = top["Probability"] * 100

    if second:
        second_percent = second["Probability"] * 100
        gap = top_percent - second_percent
        return (
            f"The model selected {top['Class']} because it had the highest confidence "
            f"at {top_percent:.2f}%. The next closest class was {second['Class']} at "
            f"{second_percent:.2f}%, giving a {gap:.2f} percentage point margin. "
            "A wider margin usually means the image matched the learned visual patterns "
            "for the top class more strongly than the alternatives."
        )

    return f"The model selected {top['Class']} with {top_percent:.2f}% confidence."


def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()


def get_page():
    if "page" not in st.session_state:
        st.session_state.page = "Classify"
    return st.session_state.page


def reset_upload():
    st.session_state.upload_key = st.session_state.get("upload_key", 0) + 1
    st.session_state.latest_result = None
    st.rerun()


def render_styles():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #faf8f5;
                --surface: #ffffff;
                --surface-soft: #f4f0ea;
                --ink: #2b2118;
                --muted: #8a7566;
                --line: #ece4da;
                --accent: #ff6b4a;
                --accent-dark: #c0432a;
                --accent-soft: #ffe7dc;
                --shadow: 0 18px 40px rgba(43, 33, 24, 0.10);
                --shadow-soft: 0 8px 20px rgba(43, 33, 24, 0.07);
            }

            html, body, [class*="css"] {
                font-family: 'Plus Jakarta Sans', sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at 14% 4%, rgba(255, 107, 74, 0.08), transparent 30rem),
                    var(--bg);
                color: var(--ink);
            }

            [data-testid="stHeader"] { background: transparent; }

            .block-container {
                max-width: 1180px;
                padding-top: 1.2rem;
                padding-bottom: 3rem;
            }

            .app-logo {
                align-items: center;
                color: var(--ink);
                display: flex;
                font-family: 'Baloo 2', sans-serif;
                font-size: 1.15rem;
                font-weight: 800;
                gap: 0.65rem;
                letter-spacing: 0;
                min-height: 3rem;
            }

            .app-mark {
                align-items: center;
                background: var(--accent);
                border-radius: 12px;
                box-shadow: var(--shadow-soft);
                color: #fff;
                display: inline-flex;
                font-family: 'Baloo 2', sans-serif;
                font-size: 0.9rem;
                font-weight: 800;
                height: 2.25rem;
                justify-content: center;
                width: 2.25rem;
            }

            .hero {
                background: var(--accent);
                border-radius: 28px;
                box-shadow: var(--shadow);
                margin: 1.1rem 0 1.4rem;
                overflow: hidden;
                padding: 2.1rem 2.2rem 2rem;
                position: relative;
            }

            .hero::after {
                content: "";
                position: absolute;
                top: -70px;
                right: -70px;
                width: 230px;
                height: 230px;
                background: rgba(255, 255, 255, 0.16);
                border-radius: 50%;
            }

            .eyebrow {
                background: rgba(255, 255, 255, 0.22);
                border-radius: 999px;
                color: #fff;
                display: inline-block;
                font-size: 0.76rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                margin-bottom: 0.9rem;
                padding: 0.32rem 0.8rem;
                position: relative;
                text-transform: uppercase;
                z-index: 1;
            }

            .hero h1 {
                color: #fff;
                font-family: 'Baloo 2', sans-serif;
                font-size: clamp(2rem, 4vw, 3.2rem);
                font-weight: 700;
                letter-spacing: 0;
                line-height: 1.08;
                margin: 0;
                max-width: 640px;
                position: relative;
                z-index: 1;
            }

            .hero p {
                color: rgba(255, 255, 255, 0.92);
                font-size: 1rem;
                line-height: 1.6;
                margin: 0.8rem 0 0;
                max-width: 640px;
                position: relative;
                z-index: 1;
            }

            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin: 1.3rem 0 0.3rem;
            }

            .badge {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 999px;
                box-shadow: var(--shadow-soft);
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 700;
                padding: 0.36rem 0.78rem;
            }

            .class-strip {
                display: grid;
                gap: 1rem;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                margin: 1.3rem 0 1.6rem;
            }

            .class-card {
                background: var(--surface);
                border-radius: 22px;
                box-shadow: var(--shadow-soft);
                min-width: 0;
                padding: 0.7rem 0.7rem 0.85rem;
                transition: transform 180ms ease, box-shadow 180ms ease;
            }

            .class-card:hover {
                box-shadow: var(--shadow);
                transform: translateY(-4px);
            }

            .class-card img {
                aspect-ratio: 1 / 1;
                border-radius: 16px;
                display: block;
                filter: drop-shadow(0 8px 10px rgba(43, 33, 24, 0.16));
                height: auto;
                object-fit: cover;
                width: 100%;
            }

            .class-card span {
                color: var(--ink);
                display: block;
                font-family: 'Baloo 2', sans-serif;
                font-size: 0.9rem;
                font-weight: 700;
                overflow: hidden;
                padding: 0.85rem 0.2rem 0;
                text-align: center;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .panel {
                background: rgba(255, 255, 255, 0.94);
                border-radius: 18px;
                box-shadow: var(--shadow);
                padding: 1.1rem;
            }

            .panel-title {
                color: var(--ink);
                font-family: 'Baloo 2', sans-serif;
                font-size: 1.05rem;
                font-weight: 700;
                margin: 0 0 0.85rem;
            }

            [data-testid="stFileUploader"] {
                background: var(--surface);
                border: 1.5px dashed #f0b7a4;
                border-radius: 16px;
                box-shadow: inset 0 0 0 9999px rgba(255, 107, 74, 0.02);
                padding: 1rem;
                transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
            }

            [data-testid="stFileUploader"]:hover {
                border-color: var(--accent);
                box-shadow: inset 0 0 0 9999px rgba(255, 107, 74, 0.06), var(--shadow-soft);
                transform: translateY(-1px);
            }

            .preview-shell img {
                border-radius: 16px;
                box-shadow: var(--shadow-soft);
            }

            .result-panel {
                background: var(--accent-dark);
                border-radius: 18px;
                box-shadow: var(--shadow);
                color: #fff;
                padding: 1.3rem 1.3rem 1.35rem;
                margin-bottom: 1.2rem;
            }

            .result-panel .label {
                color: rgba(255, 255, 255, 0.7);
                font-size: 0.76rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .result-panel h2 {
                color: #fff;
                font-family: 'Baloo 2', sans-serif;
                font-size: clamp(2rem, 4vw, 3.1rem);
                font-weight: 700;
                letter-spacing: 0;
                line-height: 1.05;
                margin: 0.3rem 0 0.45rem;
            }

            .result-panel p {
                color: rgba(255, 255, 255, 0.78);
                margin: 0;
            }

            .empty-state {
                align-items: center;
                background: var(--surface);
                border-radius: 18px;
                box-shadow: var(--shadow-soft);
                color: var(--muted);
                display: flex;
                min-height: 17rem;
                padding: 1.2rem;
            }

            .empty-state strong {
                color: var(--ink);
                display: block;
                font-family: 'Baloo 2', sans-serif;
                font-size: 1.1rem;
                margin-bottom: 0.32rem;
            }

            .probability-list {
                display: grid;
                gap: 0.85rem;
                margin: 1.1rem 0;
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
                font-weight: 750;
            }

            .probability-label strong {
                color: var(--accent-dark);
                font-size: 0.88rem;
                font-weight: 800;
            }

            .probability-track {
                background: var(--line);
                border-radius: 999px;
                height: 0.72rem;
                overflow: hidden;
            }

            .probability-fill {
                animation: fillBar 780ms cubic-bezier(.2, .8, .2, 1) both;
                background: var(--accent);
                border-radius: 999px;
                height: 100%;
            }

            .probability-fill.high { opacity: 1; }
            .probability-fill.medium { opacity: 0.62; }
            .probability-fill.low { opacity: 0.32; }

            @keyframes fillBar {
                from { width: 0; }
            }

            .explanation-box {
                background: var(--surface-soft);
                border-radius: 14px;
                color: var(--ink);
                line-height: 1.55;
                margin-top: 0.85rem;
                padding: 0.95rem 1.05rem;
            }

            .history-card {
                background: var(--surface);
                border-radius: 18px;
                box-shadow: var(--shadow-soft);
                margin-bottom: 0.8rem;
                padding: 0.85rem;
                transition: box-shadow 180ms ease, transform 180ms ease;
            }

            .history-card:hover {
                box-shadow: var(--shadow);
                transform: translateY(-2px);
            }

            .history-image-shell img {
                width: 100%;
                height: auto;
                border-radius: 14px;
                object-fit: cover;
                display: block;
                background: transparent;
                filter: drop-shadow(0 6px 8px rgba(43, 33, 24, 0.14));
            }

            .history-grid {
                display: grid;
                grid-template-columns: 110px 1fr 120px;
                gap: 1rem;
                align-items: center;
            }

            .history-meta {
                color: var(--muted);
                font-size: 0.82rem;
                line-height: 1.45;
            }

            .history-image-label {
                display: block;
                text-align: left;
                margin-top: 0.45rem;
                color: var(--ink);
                font-family: 'Baloo 2', sans-serif;
                font-weight: 700;
            }

            .history-score {
                background: var(--surface-soft);
                border-radius: 14px;
                padding: 0.55rem 0.7rem;
                text-align: center;
            }

            .history-score .value {
                color: var(--accent-dark);
                font-weight: 800;
                font-size: 1.05rem;
            }

            .history-title {
                color: var(--ink);
                font-family: 'Baloo 2', sans-serif;
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }

            .stButton > button {
                border-radius: 14px;
                font-weight: 700;
                min-height: 2.8rem;
                padding: 0.55rem 1rem;
                transition: background-color 160ms ease, border-color 160ms ease, color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
                width: 100%;
            }

            .stButton > button:hover {
                box-shadow: var(--shadow-soft);
                transform: translateY(-1px) scale(1.01);
            }

            .stButton > button[kind="primary"] {
                background: var(--accent);
                border-color: var(--accent);
                color: #ffffff;
            }

            .stButton > button[data-testid="stBaseButton-primary"] {
                background: var(--accent);
                border-color: var(--accent);
                color: #ffffff;
            }

            .stButton > button[kind="primary"]:hover {
                background: var(--accent-dark);
                border-color: var(--accent-dark);
                color: #ffffff;
            }

            .stButton > button[data-testid="stBaseButton-primary"]:hover {
                background: var(--accent-dark);
                border-color: var(--accent-dark);
                color: #ffffff;
            }

            .stButton > button[kind="secondary"] {
                background: #ffffff;
                border-color: var(--line);
                color: var(--ink);
            }

            .stButton > button[data-testid="stBaseButton-secondary"] {
                background: #ffffff;
                border-color: var(--line);
                color: var(--ink);
            }

            .stButton > button[kind="secondary"]:hover {
                background: var(--accent-soft);
                border-color: var(--accent);
                color: var(--accent-dark);
            }

            .stButton > button[data-testid="stBaseButton-secondary"]:hover {
                background: var(--accent-soft);
                border-color: var(--accent);
                color: var(--accent-dark);
            }

            div[data-testid="stMetric"] {
                background: var(--surface);
                border-radius: 14px;
                box-shadow: var(--shadow-soft);
                padding: 0.7rem 0.85rem;
            }

            [data-testid="stMetricValue"] {
                color: var(--accent-dark);
                font-weight: 800;
            }

            @media (max-width: 860px) {
                .class-strip {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }

                .class-card:last-child {
                    grid-column: span 2;
                }
            }

            @media (max-width: 620px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                .hero {
                    padding: 1.6rem 1.4rem 1.5rem;
                }

                .class-strip {
                    grid-template-columns: 1fr;
                }

                .class-card:last-child {
                    grid-column: auto;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_navbar():
    page = get_page()
    logo_col, classify_col, history_col = st.columns([6.2, 1.25, 1.25])

    with logo_col:
        st.markdown(
            """
            <div class="app-logo">
                <span class="app-mark">FC</span>
                <span>Food Image Classifier</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with classify_col:
        if st.button(
            "Classify",
            use_container_width=True,
            type="primary" if page == "Classify" else "secondary",
            help="Open the image classification page.",
        ):
            set_page("Classify")

    with history_col:
        if st.button(
            "History",
            use_container_width=True,
            type="primary" if page == "History" else "secondary",
            help="Open saved prediction history.",
        ):
            set_page("History")


def render_class_cards(class_names, input_width, input_height):
    cards = []
    assets_dir = Path("assets/classes")
    for class_name in class_display_order(class_names):
        # Prefer cleaned/edited class images in assets, fallback to representative image
        asset_image = assets_dir / f"{class_name}.png"
        if asset_image.exists():
            image_path = asset_image
        else:
            image_path = representative_image(class_name)

        image_uri = image_to_data_uri(image_path) if image_path else ""
        image_markup = (
            f'<img src="{image_uri}" alt="{format_label(class_name)}" title="{format_label(class_name)}">'
            if image_uri
            else ""
        )

        cards.append(
            (
                '<div class="class-card" style="display:flex;flex-direction:column;align-items:center;gap:0.5rem;">'
                f'{image_markup}'
                f'<span>{format_label(class_name)}</span>'
                '</div>'
            )
        )

    st.markdown(
        (
            '<div class="badge-row">'
            f'<span class="badge">{input_width} x {input_height} input</span>'
            '<span class="badge">MobileNetV2 transfer learning</span>'
            '<span class="badge">Prediction history saved separately</span>'
            '</div>'
            f'<div class="class-strip">{"".join(cards)}</div>'
        ),
        unsafe_allow_html=True,
    )


def render_probability_bars(rows):
    bars = []
    for row in rows:
        percent = row["Probability"] * 100
        tier = confidence_tier(row["Probability"])
        bars.append(
            '<div class="probability-row">'
            '<div class="probability-label">'
            f'<span>{row["Class"]}</span>'
            f"<strong>{percent:.2f}%</strong>"
            "</div>"
            '<div class="probability-track">'
            f'<div class="probability-fill {tier}" style="width: {percent:.2f}%;"></div>'
            "</div>"
            "</div>"
        )

    st.markdown(
        f'<div class="probability-list">{"".join(bars)}</div>',
        unsafe_allow_html=True,
    )


def render_empty_prediction():
    st.markdown(
        """
        <div class="empty-state">
            <div>
                <strong>Upload an image to see predictions</strong>
                Pick a JPG or PNG food photo, then run the classifier to see the top class,
                confidence breakdown, and optional text explanation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result):
    st.markdown(
        f"""
        <div class="result-panel">
            <div class="label">Top prediction</div>
            <h2>{result["predicted_label"]}</h2>
            <p>{result["confidence"] * 100:.2f}% model confidence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_left, metric_right = st.columns(2)
    metric_left.metric("Confidence", f'{result["confidence"] * 100:.2f}%')
    metric_right.metric("Alternatives", f'{max(len(result["rows"]) - 1, 0)}')

    render_probability_bars(result["rows"])

    if result["show_explanation"]:
        st.markdown(
            f'<div class="explanation-box">{prediction_reason(result["rows"])}</div>',
            unsafe_allow_html=True,
        )

def render_classify_page(model, class_names, input_width, input_height):
    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Food Recognition</div>
            <h1>Classify a food photo</h1>
            <p>
                Upload one image and the trained model will identify whether it looks like
                fried rice, sushi, ice cream, hamburger, or cheesecake.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_class_cards(class_names, input_width, input_height)

    upload_column, result_column = st.columns([1.05, 0.95], gap="large")

    with upload_column:
        st.markdown('<div class="panel-title">Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"],
            key=f"image_upload_{st.session_state.get('upload_key', 0)}",
        )

        image = None
        image_bytes = None
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            upload_signature = hashlib.sha256(image_bytes).hexdigest()
            if st.session_state.get("current_upload_signature") != upload_signature:
                st.session_state.current_upload_signature = upload_signature
                st.session_state.latest_result = None

            image = Image.open(BytesIO(image_bytes))
            st.markdown('<div class="preview-shell">', unsafe_allow_html=True)
            st.image(image, caption=uploaded_file.name, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            remove_col, replace_col = st.columns([1, 1])
            with remove_col:
                if st.button("Upload New", use_container_width=True, help="Clear image and choose another file."):
                    reset_upload()
            with replace_col:
                st.caption("Drop another file above to replace it.")
        else:
            st.session_state.current_upload_signature = None
            st.info("Upload or drag a food image into the dropzone.")

        explain_prediction = st.checkbox("Show text explanation", value=True, disabled=image is None)
        predict_clicked = st.button(
            "Classify Image",
            disabled=image is None,
            type="primary",
            use_container_width=True,
            help="Run the model on the uploaded image.",
        )

        if st.button(
            "View Prediction History",
            use_container_width=True,
            help="Open the saved prediction database.",
        ):
            set_page("History")

    with result_column:
        st.markdown('<div class="panel-title">Prediction</div>', unsafe_allow_html=True)

        if predict_clicked and image is not None:
            placeholder = st.empty()
            placeholder.markdown(
                """
                <div class="empty-state">
                    <div>
                        <strong>Analyzing image...</strong>
                        Running inference and saving the prediction record.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.spinner("Running model inference..."):
                input_image = prepare_image(image, (input_width, input_height))
                predictions = model.predict(input_image, verbose=0)[0]
                predicted_index = int(np.argmax(predictions))
                confidence = float(predictions[predicted_index])
                rows = probability_rows(class_names, predictions)
                saved_image_path = save_uploaded_image(uploaded_file.name, image_bytes)
                saved_prediction_id = save_prediction(
                    saved_image_path,
                    class_names[predicted_index],
                    confidence,
                )

            result = {
                "predicted_label": format_label(class_names[predicted_index]),
                "confidence": confidence,
                "rows": rows,
                "saved_prediction_id": saved_prediction_id,
                "show_explanation": explain_prediction,
            }
            st.session_state.latest_result = result
            placeholder.empty()
            render_result(result)
        elif st.session_state.get("latest_result"):
            st.session_state.latest_result["show_explanation"] = explain_prediction
            render_result(st.session_state.latest_result)
        else:
            render_empty_prediction()


def filtered_history_rows(class_names):
    rows = fetch_predictions()
    selected_class = st.session_state.get("history_class_filter", "All classes")
    sort_order = st.session_state.get("history_sort", "Newest first")

    if selected_class != "All classes":
        raw_class = next(
            (class_name for class_name in class_names if format_label(class_name) == selected_class),
            None,
        )
        rows = [row for row in rows if row["predicted_class"] == raw_class]

    rows = sorted(
        rows,
        key=lambda row: row["prediction_datetime"],
        reverse=sort_order == "Newest first",
    )
    return rows


def render_history_card(row):
    image_path = Path(row["uploaded_image_path"])
    image_uri = ""
    if image_path.exists():
        image_uri = image_to_data_uri(image_path) or ""

    label = format_label(row["predicted_class"])
    datetime = row.get("prediction_datetime", "")
    uploaded_path = row.get("uploaded_image_path", "")
    confidence_pct = f"{row.get('confidence_score', 0) * 100:.2f}%"

    inner_html = (
        '<div class="history-grid">'
        f'<div>'
        f'{f"<div class=\"history-image-shell\"><img src=\"{image_uri}\" alt=\"{label}\"></div>" if image_uri else "<div class=\"history-image-shell\">Image missing</div>"}'
        f'<span class="history-image-label">{label}</span>'
        '</div>'
        f'<div>'
        f'<div class="history-title">{label}</div>'
        f'<div class="history-meta">Saved at {datetime}</div>'
        '</div>'
        f'<div><div class="history-score"><div class="value">{confidence_pct}</div><div style="font-size:0.78rem;color:var(--muted);margin-top:0.18rem;">Confidence</div></div></div>'
        '</div>'
    )

    # Wrapper div + inner content are rendered in a single st.markdown call so
    # the .history-card div actually wraps its contents in the DOM (Streamlit
    # renders every st.markdown() call as its own isolated element, so
    # splitting the opening/closing tags across calls produced an empty,
    # separately-styled "ghost card" above each real card).
    st.markdown(f'<div class="history-card">{inner_html}</div>', unsafe_allow_html=True)


def render_history_page(class_names):
    st.markdown(
        """
        <section class="hero">
            <h1>History</h1>
            <p>
                Browse saved predictions from the local SQLite database. These records are
                audit/history data only, not training data.
            </p>
            <div class="hero-emoji"></div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    filter_col, sort_col, count_col = st.columns([2, 1.35, 1])
    with filter_col:
        st.selectbox(
            "Filter by predicted class",
            ["All classes"] + [format_label(class_name) for class_name in class_display_order(class_names)],
            key="history_class_filter",
        )
    with sort_col:
        st.radio("Sort by date", ["Newest first", "Oldest first"], horizontal=True, key="history_sort")

    rows = filtered_history_rows(class_names)
    all_rows = fetch_predictions()

    if "confirm_delete_all" not in st.session_state:
        st.session_state.confirm_delete_all = False

    if all_rows:
        if st.session_state.get("confirm_delete_all"):
            st.warning("This will permanently delete every saved prediction record from the SQLite database. This cannot be undone.")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button(
                    "Confirm Delete All",
                    type="primary",
                    use_container_width=True,
                    help="Permanently delete all prediction history.",
                ):
                    clear_predictions()
                    st.session_state.confirm_delete_all = False
                    st.session_state.history_page = 1
                    st.success("Prediction history cleared.")
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", use_container_width=True, help="Cancel deleting all history."):
                    st.session_state.confirm_delete_all = False
                    st.rerun()
        else:
            if st.button(
                "Delete All History",
                type="secondary",
                use_container_width=True,
                help="Permanently remove every saved prediction record.",
            ):
                st.session_state.confirm_delete_all = True
                st.rerun()

    with count_col:
        st.metric("Records", len(rows))

    if not rows:
        st.markdown(
            """
            <div class="empty-state">
                <div>
                    <strong>No predictions found</strong>
                    Try a different filter or classify a new image first.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    total_pages = max(1, math.ceil(len(rows) / PAGE_SIZE))
    current_page = min(st.session_state.get("history_page", 1), total_pages)
    st.session_state.history_page = current_page

    start = (current_page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE

    for row in rows[start:end]:
        render_history_card(row)

    prev_col, page_col, next_col = st.columns([1, 2, 1])
    with prev_col:
        if st.button(
            "Previous",
            disabled=current_page <= 1,
            use_container_width=True,
            help="Show the previous page of prediction history.",
        ):
            st.session_state.history_page = current_page - 1
            st.rerun()
    with page_col:
        st.markdown(
            f"<div style='text-align:center;color:#8a7566;font-weight:750;'>Page {current_page} of {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with next_col:
        if st.button(
            "Next",
            disabled=current_page >= total_pages,
            use_container_width=True,
            help="Show the next page of prediction history.",
        ):
            st.session_state.history_page = current_page + 1
            st.rerun()

    # if st.button("Clear all predictions", use_container_width=True):
    #     clear_predictions()
    #     st.success("All predictions cleared.")
    #     st.rerun()


st.set_page_config(
    page_title="Food Classifier",
    page_icon=":fork_and_knife:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_styles()
render_navbar()

if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    st.error("Model files were not found.")
    st.info("Train the model first by running `python train.py`, then reopen this app.")
    st.stop()

init_database()
model = load_model()
class_names = load_class_names()
input_height = model.input_shape[1]
input_width = model.input_shape[2]

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if get_page() == "History":
    render_history_page(class_names)
else:
    render_classify_page(model, class_names, input_width, input_height)