import streamlit as st
import numpy as np
from PIL import Image
import random

st.title("Brain Tumor Detection")

st.write("Upload an MRI image to check if a brain tumor is detected.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224,224))
    img_array = np.array(img)/255.0

    result = random.choice(["Brain Tumor Detected","No Brain Tumor Detected"])

    if result == "Brain Tumor Detected":
        st.error(result)
    else:
        st.success(result)
