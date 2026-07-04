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


st.set_page_config(page_title="Food Classifier", page_icon="🍽️", layout="centered")

st.title("Food Image Classifier")
st.write("Upload a food image and the CNN model will predict one of the trained classes.")

if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    st.warning(
        "Model files were not found. Train the model first by running `python train.py`."
    )
    st.stop()

model = load_model()
class_names = load_class_names()
input_height = model.input_shape[1]
input_width = model.input_shape[2]

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Selected image", use_container_width=True)

    if st.button("Predict"):
        input_image = prepare_image(image, (input_width, input_height))
        predictions = model.predict(input_image, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])

        st.subheader(format_label(class_names[predicted_index]))
        st.metric("Confidence", f"{confidence * 100:.2f}%")

        st.write("Class probabilities")
        probability_rows = []
        for class_name, probability in zip(class_names, predictions):
            probability_rows.append(
                {
                    "Class": format_label(class_name),
                    "Probability": round(float(probability), 4),
                }
            )
        st.dataframe(probability_rows, hide_index=True, use_container_width=True)
