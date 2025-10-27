🧠 AI vs. Real Image Classifier

Et maskinlæringsprosjekt som klassifiserer ansiktsbilder som enten AI-genererte eller ekte ved hjelp av et nevralt nettverk. Modellen trenes med transfer learning (ResNet18) og gjøres tilgjengelig via en interaktiv webapplikasjon i Streamlit.

📌 Innhold

Introduksjon

Datasett

Metode

Modellarkitektur

Installasjon

Kjøring

Resultater

Webapplikasjon

Mappe-struktur

Videre arbeid

Forfattere

🚀 Introduksjon

Formålet med prosjektet er å undersøke om det er mulig å skille AI-genererte ansikter fra ekte bilder.
Dette er en relevant problemstilling med økende spredning av deepfakes, generative modeller og manipulerte bilder.

Prosjektet demonstrerer:

✅ maskinlæringens komplette livssyklus
✅ databehandling og preprocessing
✅ trening og evaluering
✅ deploy i en webapplikasjon

📂 Datasett

Datasettet består av to kategorier:

real – ekte ansiktsbilder

ai – AI-genererte ansikter

Datasettet ble delt i:

70 % trening

15 % validering

15 % testing

Data ble lagret i:

data/processed/train/
data/processed/val/
data/processed/test/

⚙️ Metode

Transfer learning med ResNet18

Fine-tuning på siste lag

Normalisering og resize med torchvision transforms

CrossEntropyLoss

Adam Optimizer

Modellen ble trent i 6 epoker på ca. 40 % av treningssettet for å redusere treningstid.

🧬 Modellarkitektur

Tilpasset ResNet18:

model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Softmax(dim=1)
)
📊 Resultater

Etter trening oppnådde modellen:

Test accuracy: 69.58 %

Klassifikasjonsrapport viste balansert F1-score:

Klasse	F1-score
AI	0.70
Real	0.69
🔷 Confusion Matrix

Modellen feiltolker oftere ekte bilder som AI, sannsynligvis fordi moderne filtre gir glattere teksturer som ligner AI-genererte ansikter.

🌐 Webapplikasjon

Webappen er laget i Streamlit.
Brukeren kan:

✅ laste opp bilder
✅ få sanntidsprediksjon
✅ se sannsynligheter
✅ motta visual feeback (ballonger/warning)

Kun det siste laget ble trent videre (fine-tuning).



📁 Mappe-struktur
ML_project/
│─ data/
│   └─ processed/
│─ src/
│   ├─ train.py
│   ├─ predict.py
│   └─ prepare_data.py
│─ webapp/
│   └─ app.py
│─ notebooks/
│─ model.pth
│─ requirements.txt
│─ README.md


Observasjoner

- Modellen er generelt stabil

- Ingen tydelig overfitting

- Ekte bilder med filtre misklassifiseres oftere

Forfattere

Sarah S. Ahsan

Amna Zafar

Mannat Gabria

DAT158 — Maskinlæring prosjekt
Høgskulen på Vestlandet (HVL)
