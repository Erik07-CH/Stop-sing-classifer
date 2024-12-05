import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps

st.title('Clasificador de Señales de Stop')

@st.cache(allow_output_mutation=True)
def load_my_model():
    model = load_model('stop_sign_classifier.h5')
    return model

model = load_my_model()

st.write("""
         ## Sube una imagen para clasificarla
         """)

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Cargar la imagen
    img = Image.open(uploaded_file)
    img_resized = img.resize((64, 64))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar la predicción
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        st.write("**No es una señal de Stop**")
    else:
        st.write("**¡Es una señal de Stop!**")

    st.image(img, caption='Imagen subida', use_column_width=True)
