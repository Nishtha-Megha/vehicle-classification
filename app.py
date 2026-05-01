import streamlit as st
import numpy as np
from tensorflow.keras.models import load_modelfrom PIL import Image
import os

# Hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Page config
st.set_page_config(page_title="Vehicle Classifier", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: white;'>🔍 Vehicle Classifier</h1>", unsafe_allow_html=True)
st.write("Upload an image to classify vehicle type 🚗 🏍️ 🚛")

# Load model
@st.cache_resource
def load_my_model():
    return load_model("vehicle_model (1).h5")

model = load_my_model()

# ⚠️ Try this order (adjust if needed)
classes = ["bike 🏍️", "car 🚗", "truck 🚛"]

# Upload
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Show image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

        # Preprocess
        img_resized = img.resize((150,150))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        with st.spinner("🔍 Analyzing image..."):
            prediction = model.predict(img_array)

        index = np.argmax(prediction)
        confidence = np.max(prediction)

        # Result
        if index < len(classes):
            result = classes[index]
        else:
            result = "Unknown ❓"

        st.success(f"✅ Prediction: {result}")
        st.info(f"📊 Confidence: {confidence*100:.2f}%")

        # Chart
        st.subheader("📊 Prediction Probabilities")
        for i, cls in enumerate(classes):
            st.write(f"{cls}: {prediction[0][i]*100:.2f}%")

        # Debug (you can remove later)
        st.write("🔎 Raw prediction:", prediction)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
