import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.title("Brain Tumor Detection")

model = load_model("BrainTumor.h5")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img)

    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.write("Brain Tumor Detected")
    else:
        st.write("No Brain Tumor Detected")
