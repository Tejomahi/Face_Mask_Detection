import streamlit as st
import numpy as np
from PIL import Image
import keras
from huggingface_hub import hf_hub_download

# Download trained model from Hugging Face
model_path = hf_hub_download(
    repo_id="Tejomahi/face-mask-detection",
    filename="face_mask_model.keras"
)

# Load trained model
model = keras.models.load_model(model_path)

st.title("😷 Face Mask Detection")

st.write(
    "Upload an image to check whether the person is wearing a mask."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image_resized = image.resize((128, 128))

    image_array = np.array(image_resized)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 128, 128, 3)

    prediction = model.predict(image_array, verbose=0)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    if predicted_class == 1:
        st.success(
            f"😷 Wearing Mask — Confidence: {confidence:.2f}%"
        )
    else:
        st.error(
            f"❌ Not Wearing Mask — Confidence: {confidence:.2f}%"
        )