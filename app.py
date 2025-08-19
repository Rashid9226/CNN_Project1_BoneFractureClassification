import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import json

# Load model and class names
model = load_model("bone_break_classification_model.keras")
with open("class_names.json", "r") as f:
    class_names = json.load(f)

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

st.title("Bone Fracture Type Classifier")
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("Prediction:")

    st.write(predicted_class)
