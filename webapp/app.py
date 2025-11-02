# app.py - ultimate proffversjon
import streamlit as st
from PIL import Image
import os
import sys
from io import BytesIO
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from predict import load_model, predict

model_path = os.path.join(os.path.dirname(__file__), '../model.pth')

st.set_page_config(
    page_title="AI vs Real Image Classifier",
    page_icon="🧠",
    layout="wide"
)

# === Custom font og CSS ===
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=American+Typewriter&display=swap');

    html, body, [class*="css"]  {
        font-family: 'American Typewriter', monospace;
        color: #333333;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    .fade-in {
        animation: fadeIn 1s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }

    .progress-bar {
        height: 25px;
        border-radius: 12px;
        background-color: #D3D3D3;
        margin-bottom: 10px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        text-align: center;
        line-height: 25px;
        color: white;
        font-weight: bold;
        transition: width 1s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Header ===
st.markdown(
    "<h1 style='text-align: center; color:#2F4F4F;'>🧠 AI vs Real Image Classifier</h1>"
    "<p style='text-align: center; font-size:18px; color:#555555;'>Last opp et bilde og se om det er AI-generert eller ekte!</p>",
    unsafe_allow_html=True
)

@st.cache_resource
def load_model_once():
    return load_model(model_path)

model = load_model_once()

st.markdown("<hr style='border:1px solid #D3D3D3'>", unsafe_allow_html=True)

# === Upload bilde ===
st.subheader("1️⃣ Last opp et bilde")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

def show_progress_bar(label, value, color):
    st.markdown(f"<div class='progress-bar'><div class='progress-fill' style='width:{value}%;background-color:{color};'>{label} {value:.1f}%</div></div>", unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        image_bytes = BytesIO(uploaded_file.read())
        image = Image.open(image_bytes).convert("RGB")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Lastet bilde", use_container_width=True)

        with col2:
            st.markdown("<h3 style='color:#2E8B57;'>2️⃣ Prediksjon</h3>", unsafe_allow_html=True)
            if st.button("🔍 Prediker bilde"):
                with st.spinner("Analyserer bildet..."):
                    result = predict(image_bytes, model)

                predicted_class = result["class"]
                probability = result["probability"]
                scores = result["scores"]

                # Toppklasse-resultat med fade-in og farge
                color = "#FF8C00" if predicted_class=="ai" else "#228B22"
                emoji = "⚠️" if predicted_class=="ai" else "✅"
                st.markdown(f"<h2 class='fade-in' style='color:{color};'>{emoji} {predicted_class.upper()}</h2>", unsafe_allow_html=True)
                if predicted_class == "real":
                    st.balloons()

                # Progress bar for toppklasse
                show_progress_bar(predicted_class.upper(), probability*100, color)

                # Detaljer per klasse med progress bars
                st.markdown("<h4 style='color:#2F4F4F;'>3️⃣ Sannsynligheter per klasse</h4>", unsafe_allow_html=True)
                for cls, score in scores.items():
                    cls_emoji = "🤖" if cls=="ai" else "👤"
                    cls_color = "#FF8C00" if cls=="ai" else "#228B22"
                    show_progress_bar(f"{cls_emoji} {cls.upper()}", score*100, cls_color)

    except Exception as e:
        st.error(f"Noe gikk galt med bildet: {e}")

st.markdown("<hr style='border:1px solid #D3D3D3'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color:#555555;'>📚 Prosjekt av Sarah S. Ahsan, Amna Zafar og Mannat Gabria</p>"
    "<p style='text-align: center; color:#555555;'>💡 Maskinlærings-app for å skille AI-genererte bilder fra ekte bilder</p>",
    unsafe_allow_html=True
)


