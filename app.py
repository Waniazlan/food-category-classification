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
        <style>
            :root {
                --bg: #f4f7f2;
                --surface: #ffffff;
                --surface-soft: #eef5ef;
                --ink: #17211c;
                --muted: #66736d;
                --line: #dbe4dd;
                --accent: #0f7b63;
                --accent-dark: #073f35;
                --accent-soft: #dff2ea;
                --yellow: #d69b17;
                --graybar: #9aa5a1;
                --shadow: 0 16px 40px rgba(25, 45, 36, 0.10);
                --shadow-soft: 0 8px 22px rgba(25, 45, 36, 0.08);
            }

            .stApp {
                background:
                    radial-gradient(circle at 12% 8%, rgba(15, 123, 99, 0.12), transparent 24rem),
                    linear-gradient(180deg, #fbfcf7 0%, var(--bg) 100%);
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
                font-size: 1.08rem;
                font-weight: 850;
                gap: 0.65rem;
                letter-spacing: 0;
                min-height: 3rem;
            }

            .app-mark {
                align-items: center;
                background: linear-gradient(135deg, var(--accent), #58a96f);
                border-radius: 10px;
                box-shadow: var(--shadow-soft);
                color: #fff;
                display: inline-flex;
                font-size: 0.9rem;
                font-weight: 900;
                height: 2.15rem;
                justify-content: center;
                width: 2.15rem;
            }

            .hero {
                padding: 1.35rem 0 0.9rem;
            }

            .eyebrow {
                color: var(--accent);
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                margin-bottom: 0.35rem;
                text-transform: uppercase;
            }

            .hero h1 {
                color: var(--ink);
                font-size: clamp(2rem, 4vw, 3.4rem);
                font-weight: 850;
                letter-spacing: 0;
                line-height: 1.04;
                margin: 0;
            }

            .hero p {
                color: var(--muted);
                font-size: 1rem;
                line-height: 1.6;
                margin: 0.75rem 0 0;
                max-width: 720px;
            }

            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin: 0.7rem 0 0.2rem;
            }

            .badge {
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(219, 228, 221, 0.86);
                border-radius: 999px;
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 750;
                padding: 0.34rem 0.72rem;
            }

            .class-strip {
                display: grid;
                gap: 0.8rem;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                margin: 1.15rem 0 1.45rem;
            }

            .class-card {
                background: var(--surface);
                border-radius: 20px;
                box-shadow: var(--shadow-soft);
                min-width: 0;
                padding: 0.65rem 0.65rem 0.75rem;
            }

            .class-card img {
                aspect-ratio: 1 / 1;
                border-radius: 18px;
                display: block;
                height: auto;
                object-fit: cover;
                width: 100%;
            }

            .class-card span {
                color: var(--ink);
                display: block;
                font-size: 0.86rem;
                font-weight: 800;
                overflow: hidden;
                padding: 0.78rem 0.2rem 0;
                text-align: center;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .panel {
                background: rgba(255, 255, 255, 0.92);
                border-radius: 12px;
                box-shadow: var(--shadow);
                padding: 1.05rem;
            }

            .panel-title {
                color: var(--ink);
                font-size: 1rem;
                font-weight: 850;
                margin: 0 0 0.85rem;
            }

            [data-testid="stFileUploader"] {
                background: var(--surface);
                border: 1.5px dashed #aec2b8;
                border-radius: 12px;
                box-shadow: inset 0 0 0 9999px rgba(15, 123, 99, 0.018);
                padding: 1rem;
                transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
            }

            [data-testid="stFileUploader"]:hover {
                border-color: var(--accent);
                box-shadow: inset 0 0 0 9999px rgba(15, 123, 99, 0.05), var(--shadow-soft);
                transform: translateY(-1px);
            }

            .preview-shell img {
                border-radius: 10px;
                box-shadow: var(--shadow-soft);
            }

            .result-panel {
                background: linear-gradient(135deg, #0a2d27, var(--accent-dark));
                border-radius: 12px;
                box-shadow: var(--shadow);
                color: #fff;
                padding: 1.25rem;
            }

            .result-panel .label {
                color: rgba(255, 255, 255, 0.68);
                font-size: 0.76rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .result-panel h2 {
                color: #fff;
                font-size: clamp(2rem, 4vw, 3.25rem);
                font-weight: 900;
                letter-spacing: 0;
                line-height: 1.02;
                margin: 0.28rem 0 0.45rem;
            }

            .result-panel p {
                color: rgba(255, 255, 255, 0.74);
                margin: 0;
            }

            .empty-state {
                align-items: center;
                background: var(--surface);
                border-radius: 12px;
                box-shadow: var(--shadow-soft);
                color: var(--muted);
                display: flex;
                min-height: 17rem;
                padding: 1.2rem;
            }

            .empty-state strong {
                color: var(--ink);
                display: block;
                font-size: 1.08rem;
                margin-bottom: 0.32rem;
            }

            .probability-list {
                display: grid;
                gap: 0.85rem;
                margin: 1.05rem 0;
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
                font-weight: 780;
            }

            .probability-label strong {
                color: var(--accent-dark);
                font-size: 0.88rem;
                font-weight: 850;
            }

            .probability-track {
                background: #e4ebe6;
                border-radius: 999px;
                height: 0.72rem;
                overflow: hidden;
            }

            .probability-fill {
                animation: fillBar 780ms cubic-bezier(.2, .8, .2, 1) both;
                border-radius: 999px;
                height: 100%;
            }

            .probability-fill.high { background: linear-gradient(90deg, #0f7b63, #5ebd75); }
            .probability-fill.medium { background: linear-gradient(90deg, #d69b17, #f1ca5a); }
            .probability-fill.low { background: linear-gradient(90deg, #87918d, #b5bfba); }

            @keyframes fillBar {
                from { width: 0; }
            }

            .explanation-box {
                background: var(--surface-soft);
                border-radius: 10px;
                color: var(--ink);
                line-height: 1.55;
                margin-top: 0.85rem;
                padding: 0.9rem 1rem;
            }

            .history-card {
                background: var(--surface);
                border-radius: 12px;
                box-shadow: var(--shadow-soft);
                margin-bottom: 0.8rem;
                padding: 0.8rem;
            }

            .history-meta {
                color: var(--muted);
                font-size: 0.82rem;
                line-height: 1.45;
            }

            .history-title {
                color: var(--ink);
                font-size: 1rem;
                font-weight: 850;
                margin-bottom: 0.25rem;
            }

            .stButton > button {
                border-radius: 10px;
                font-weight: 800;
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
                background: #b7352d;
                border-color: #b7352d;
                color: #ffffff;
            }

            .stButton > button[data-testid="stBaseButton-primary"] {
                background: #b7352d;
                border-color: #b7352d;
                color: #ffffff;
            }

            .stButton > button[kind="primary"]:hover {
                background: #9f2d26;
                border-color: #9f2d26;
                color: #ffffff;
            }

            .stButton > button[data-testid="stBaseButton-primary"]:hover {
                background: #9f2d26;
                border-color: #9f2d26;
                color: #ffffff;
            }

            .stButton > button[kind="secondary"] {
                background: #ffffff;
                border-color: #cbd8d1;
                color: var(--ink);
            }

            .stButton > button[data-testid="stBaseButton-secondary"] {
                background: #ffffff;
                border-color: #cbd8d1;
                color: var(--ink);
            }

            .stButton > button[kind="secondary"]:hover {
                background: #edf5f0;
                border-color: var(--accent);
                color: var(--accent-dark);
            }

            .stButton > button[data-testid="stBaseButton-secondary"]:hover {
                background: #edf5f0;
                border-color: var(--accent);
                color: var(--accent-dark);
            }

            div[data-testid="stMetric"] {
                background: var(--surface);
                border-radius: 10px;
                box-shadow: var(--shadow-soft);
                padding: 0.7rem 0.85rem;
            }

            [data-testid="stMetricValue"] {
                color: var(--accent-dark);
                font-weight: 900;
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
    for class_name in class_display_order(class_names):
        image_path = representative_image(class_name)
        image_uri = image_to_data_uri(image_path) if image_path else ""
        image_markup = f'<img src="{image_uri}" alt="{format_label(class_name)}">' if image_uri else ""
        cards.append(
            f'<div class="class-card">{image_markup}<span>{format_label(class_name)}</span></div>'
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

    st.caption(f'Saved prediction #{result["saved_prediction_id"]} to the prediction database.')
    st.caption("Saved uploads are kept for prediction history only and are not used as training data.")


def render_classify_page(model, class_names, input_width, input_height):
    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">MobileNetV2 Food Recognition</div>
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
    image_col, detail_col, score_col = st.columns([1.1, 3.2, 1.1])

    with image_col:
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.caption("Image missing")

    with detail_col:
        st.markdown(
            f"""
            <div class="history-title">{format_label(row["predicted_class"])}</div>
            <div class="history-meta">
                Saved at {row["prediction_datetime"]}<br>
                {row["uploaded_image_path"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with score_col:
        st.metric("Confidence", f'{row["confidence_score"] * 100:.2f}%')


def render_history_page(class_names):
    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Prediction Database</div>
            <h1>History</h1>
            <p>
                Browse saved predictions from the local SQLite database. These records are
                audit/history data only, not training data.
            </p>
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
        with st.container():
            st.markdown('<div class="history-card">', unsafe_allow_html=True)
            render_history_card(row)
            st.markdown("</div>", unsafe_allow_html=True)

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
            f"<div style='text-align:center;color:#66736d;font-weight:750;'>Page {current_page} of {total_pages}</div>",
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
