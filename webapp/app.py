# app.py
import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import tempfile

# Importer dine prediksjons-funksjoner fra predict.py
# predict.py må ligge i samme mappe som denne filen.
from predict import load_model, predict, CLASS_NAMES

st.set_page_config(
    page_title="AI vs Ekte - Demo",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Enkel CSS for finere layout ---
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        max-width: 880px;
        margin: 18px auto;
    }
    h1 { text-align:center; color:#1f2937; }
    .uploader { display:flex; gap:12px; align-items:center; justify-content:center; }
    .predictionBox { margin-top:18px; text-align:center; padding:14px; border-radius:10px; font-weight:600; box-shadow: 0 4px 12px rgba(0,0,0,0.04); }
    .meta { color:#6b7280; font-size:0.9rem; text-align:center; margin-top:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🧠 Bildeklassifiserer — AI vs Ekte")
st.write("Last opp et bilde, så bruker vi den trente modellen din til å predikere om bildet er **AI-generert** eller **ekte**.")

# Sidebar for modellvalg / device
st.sidebar.header("Innstillinger")
model_path = st.sidebar.text_input("Sti til model.pth", value="model.pth")
device_choice = st.sidebar.selectbox("Enhet for inferens", options=["auto", "cpu", "cuda", "mps"])
st.sidebar.markdown("Legg merke til at `cuda` fungerer kun hvis du har CUDA GPU. `mps` for Apple silicon.")

# Last modellen – behold i session slik at den ikke lastes om igjen hver gang
@st.cache_resource
def _load_model_cached(path, device_choice):
    # Gjør device override pass-through til predict.load_model
    device_override = None if device_choice == "auto" else device_choice
    model = load_model(path, device_override=device_override)
    return model

# Vis knapp for å (re)laste hvis brukeren vil
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("Velg et bilde (jpg / jpeg / png)", type=["jpg", "jpeg", "png"])
with col2:
    load_btn = st.button("Last modell")

model = None
if load_btn:
    try:
        with st.spinner("Laster modell..."):
            model = _load_model_cached(model_path, device_choice)
        st.success("Modellen er lastet.")
    except Exception as e:
        st.error(f"Kunne ikke laste modell: {e}")

# Hvis vi ikke lastet modellen via knapp, prøv automatisk (bedre UX)
if model is None:
    if os.path.exists(model_path):
        try:
            with st.spinner("Laster modell..."):
                model = _load_model_cached(model_path, device_choice)
        except Exception as e:
            st.warning("Kunne ikke laste modell automatisk. Trykk 'Last modell' i sidefeltet eller sjekk sti.")
    else:
        st.info("Modell ikke funnet lokalt — legg model.pth i prosjektmappen eller skriv korrekt sti i sidefeltet.")

# Når bilde er lastet opp
if uploaded_file is not None:
    # Vis forhåndsvisning
    try:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"Kunne ikke lese bildet: {e}")
        st.stop()

    st.image(image, caption="📸 Lastet opp bilde", use_column_width=False, width=480)

    if model is None:
        st.error("Modellen er ikke lastet. Sjekk model.pth og trykk 'Last modell'.")
    else:
        # Lagre midlertidig fil (predict.predict jobber med filsti)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path, format="PNG")

        # Kjør prediksjon og vis resultat
        try:
            with st.spinner("Kjører prediksjon..."):
                res = predict(tmp_path, model, device_override=(None if device_choice == "auto" else device_choice))
        except Exception as e:
            st.error(f"Prediksjon feilet: {e}")
            # Fjern tempfil
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        else:
            # Fjern tempfil etter bruk
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            pred_class = res.get("class", "unknown")
            prob = res.get("probability", 0.0)
            scores = res.get("scores", {})

            # Farge og bakgrunn basert på kategori
            if pred_class.lower() in ["ai", "ai-generated"]:
                color = "#b91c1c"  # rød
                bg = "#fff1f0"
                label = "AI-generert"
            else:
                color = "#065f46"  # grønn
                bg = "#ecfdf5"
                label = "Ekte"

            st.markdown(
                f"""
                <div class="predictionBox" style="background:{bg}; color:{color}">
                    🔍 Predikert: <strong>{label}</strong> — {prob*100:.2f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Vis score per klasse
            st.write("**Sannsynligheter per klasse:**")
            # Vis i tabell
            import pandas as pd
            rows = []
            for cls in CLASS_NAMES:
                rows.append({"klasse": cls, "sannsynlighet (%)": round(scores.get(cls, 0.0) * 100, 2)})
            df = pd.DataFrame(rows)
            st.table(df)

st.markdown('<div class="meta">Lag din app: plasser `predict.py` + `model.pth` i samme mappe som denne filen og kjør <code>streamlit run app.py</code></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


