# Dummy prediction (for demo)
import random

classes = ["bike 🏍️", "car 🚗", "truck 🚛"]

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Analyzing image..."):
            result = random.choice(classes)
            confidence = random.uniform(0.7, 0.95)

        st.success(f"✅ Prediction: {result}")
        st.info(f"📊 Confidence: {confidence*100:.2f}%")

    except Exception as e:
        st.error(f"❌ Error: {e}")
