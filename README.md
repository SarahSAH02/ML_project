# 🧠 AI vs. Real Image Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-69.58%25-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

## 💡 Introduksjon

Dette prosjektet undersøker om det er mulig å skille **AI-genererte** ansikter fra **ekte** bilder ved hjelp av maskinlæring.  
Modellen trenes med *transfer learning* (ResNet18) og gjøres tilgjengelig i en interaktiv Streamlit-webapp.

Prosjektet demonstrerer:

✅ maskinlæringens komplette livssyklus  
✅ databehandling og preprocessing  
✅ trening og evaluering  
✅ deployment i webapplikasjon  

---
### 🎥 Demonstrasjon av appen

![App Demo](media/app_demo.gif)

## 📌 Innhold

- [Datasett](#-datasett)
- [Metode](#-metode)
- [Modellarkitektur](#-modellarkitektur)
- [Resultater](#-resultater)
- [Webapplikasjon](#-webapplikasjon)
- [Forfattere](#-forfattere)

---

## 📂 Datasett

Datasettet består av to kategorier:

- 🟣 `ai` – AI-genererte ansikter
- 🟩 `real` – ekte ansiktsbilder

Fordeling:

- 70 % — trening  
- 15 % — validering  
- 15 % — testing  

Datasettet ble prosessert slik:

data/processed/train/
data/processed/val/
data/processed/test/



---

## ⚙️ Metode

Teknologier og teknikker brukt:

- Transfer learning via **ResNet18**
- Fine-tuning på siste lag
- Normalize + resize med torchvision transforms
- **CrossEntropyLoss** som tapsfunksjon
- **Adam** optimizer

Modellen ble trent i **6 epoker** på ca. **40 %** av datasetet for å redusere treningstiden.

---

## 🧬 Modellarkitektur

ResNet18 ble tilpasset med nytt klassifiseringslag:

```python
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Softmax(dim=1)
)
Kun siste lag ble trent videre (fine-tuning).

📊 Resultater
Endelig testaccuracy:

✅ 69.58 %

Klassifikasjonsrapporten viste balanserte resultater:

Klasse	F1-score
AI	0.70
Real	0.69

🔷 Confusion Matrix
Modellen feiltolker oftere ekte bilder som AI.
Dette kan skyldes filtre og moderne etterbehandling som gir glatte, AI-lignende teksturer.

🧐 Observasjoner
Modellen er generelt stabil

Ingen tydelig overfitting

Feil skjer oftest på ekte bilder med “kunstige” trekk

🌐 Webapplikasjon
Webappen er laget med Streamlit.
Brukeren kan:

✅ laste opp bilder
✅ få sanntidsprediksjon
✅ se sannsynligheter
✅ motta visuelt feedback (ballonger/warnings)


👩‍🦰👱‍♀️👩Forfattere

- Sarah S. Ahsan
- Amna Zafar
- Mannat Gabria
