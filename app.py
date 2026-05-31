import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

# Load trained model
model = tf.keras.models.load_model(
    "saved_model/pneumonia_model.h5"
)

# Sidebar
st.sidebar.title("About")

st.sidebar.info(
    """
    AI-based Pneumonia Detection System
    
    Technologies:
    - TensorFlow
    - CNN
    - Streamlit
    """
)

# Title
st.markdown(
    """
    <h1 style='text-align:center; color:#1E88E5;'>
    🩺 Pneumonia Detection System
    </h1>
    """,
    unsafe_allow_html=True
)

st.write(
    "Upload a Chest X-ray image to detect Pneumonia."
)

# Upload image
uploaded_file = st.file_uploader(
    "📤 Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# If image uploaded
if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        img,
        caption="Uploaded X-ray",
        use_container_width=True
    )

    # Resize image
    resized_img = img.resize((224, 224))

    # Convert to array
    img_array = image.img_to_array(resized_img)

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize
    img_array = img_array / 255.0

    # Predict button
    if st.button("🔍 Predict"):

        with st.spinner("Analyzing X-ray..."):

            time.sleep(2)

            prediction = model.predict(img_array)

            confidence = float(prediction[0][0])

            # PNEUMONIA
            if confidence > 0.5:

                pneumonia_score = confidence * 100

                st.error("⚠️ Pneumonia Detected")

                st.metric(
                    label="Prediction Confidence",
                    value=f"{pneumonia_score:.2f}%"
                )

                st.warning(
                    "Please consult a medical professional."
                )

            # NORMAL
            else:

                normal_score = (1 - confidence) * 100

                st.success("✅ Patient is NORMAL")

                st.metric(
                    label="Prediction Confidence",
                    value=f"{normal_score:.2f}%"
                )

                # Celebration balloons
                st.balloons()

                st.markdown(
                    """
                    <h2 style='text-align:center; color:green;'>
                    🎉 Great News! No Pneumonia Detected 🎉
                    </h2>
                    """,
                    unsafe_allow_html=True
                )

# Footer
st.markdown("---")

st.markdown(
    """
    <center>
    Developed with ❤️ using Streamlit & TensorFlow
    </center>
    """,
    unsafe_allow_html=True
)