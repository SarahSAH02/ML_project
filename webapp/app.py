import streamlit as st
import numpy as np
import cv2
import os

# Plassholdere for modell og prediksjonsfunksjon
# Disse vil bli implementert senere når de tilknyttede filene er klare
class model.pth:
    def predict(self, image):
        # Dummy prediksjon: Returner tilfeldig klasse
        return np.random.choice(['AI-generated', 'Real'])

# model = load_model('model.path') # Comment out until the model is ready
model = load_model('model.pth')  # Placeholder for the actual model

# Funksjon for å forbehandle bildet
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))  # Endre til riktig størrelse
    image = np.array(image) / 255.0  # Normalisere
    image = np.reshape(image, (1, 128, 128, 3))  # Endre form
    return image

# Gjør prediksjon
def predict_image(image):
    processed_image = preprocess_image(image)
    # Bruk den faktiske modellen når den er tilgjengelig
    prediction = model.pth(processed_image)  # Placeholder function
    return prediction

# Streamlit-applikasjon
st.title("Bildeklassifiserer for AI vs Ekte Bilder")
st.write("Last opp et bilde for å få en prediksjon.")

uploaded_file = st.file_uploader("Velg et bilde...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption='Lastet opp bilde.', use_column_width=True)

    prediction = predict_image(image)
    st.write(f'Prediksjon: {prediction}')



