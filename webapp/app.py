#Må laste ned "streamlit" om du ikke har: bruk kommando pip install streamlit
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Last inn modellen
model = load_model('path_to_your_model/image_classifier_model.h5')  # Tilpass stien til modellen din

def predict_image(image):
    # Forbehandle bildet for modellen
    image = cv2.resize(image, (128, 128))  # Endre størrelse til (128, 128)
    image = np.array(image) / 255.0  # Normaliser
    image = np.reshape(image, (1, 128, 128, 3))  # Endre form
    prediction = model.predict(image)

    return "AI-generated" if prediction[0] > 0.5 else "Real"

# Streamlit-app
st.title("Image Classifier")
st.write("Last opp et bilde for å få en prediksjon.")

uploaded_file = st.file_uploader("Velg et bilde...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leser og viser bildet
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption='Lastet opp bilde.', use_column_width=True)

    prediction = predict_image(image)
    st.write(f'Prediksjon: {prediction}')
