import streamlit as st
from PIL import Image
import numpy as np
import cv2

from tensorflow.keras.models import load_model


# Load model

model = load_model("saved_models\model1.h5")

def predict(image: Image.Image):
    """
    Takes a PIL image and returns (label, confidence)
    """
    # convert to numpy
    img_arr = np.array(image)

    image_resized = cv2.resize(img_arr, (128,128))
    image_exp = np.expand_dims(image_resized, axis=0) / 255.0
    pred_proba = model.predict(image_exp)[0][0]

    if pred_proba > 0.5:
        label = 'Real'
        confidence = pred_proba
    else:
        label = "Fake"
        confidence = 1 - pred_proba

    return label, round(confidence,2)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️‍♂️",
    layout="centered"
)

st.title("🕵️ Deepfake Image Detector")
st.write("Upload an image to check whether it's **Real** or **Fake**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image preview
    image = Image.open(uploaded_file).convert("RGB")
    display_img = image.resize((250, 250)) 
    st.image(display_img, caption="Uploaded Image")

    # Detect button
    if st.button("🔍 Detect"):
        with st.spinner("Analyzing image..."):
            label, confidence = predict(image)

        # Show result
        if label == "Real":
            st.success(f"✅ Prediction: {label} ({confidence*100:.1f}% confidence)")
        else:
            st.error(f"⚠️ Prediction: {label} ({confidence*100:.1f}% confidence)")


