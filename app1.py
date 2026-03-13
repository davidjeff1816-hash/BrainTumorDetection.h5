import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.title("Brain Tumor Detection")

@st.cache_resource
def load_ai_model():
    model = load_model("BrainTumor.h5")
    return model

model = load_ai_model()

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img)

    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("Brain Tumor Detected")
    else:
        st.success("No Brain Tumor Detected")
