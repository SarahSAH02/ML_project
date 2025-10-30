# predict.py
# Enkel predictor for AI vs Real bilder
#
# Viktig:
# - Denne filen bruker klasseordre A: ["ai", "real"]
# - Transform og normalisering er satt til nøyaktig det som står i train.py
# - For å endre klasseorden, bytt rekkefølgen i CLASS_NAMES

import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# === Enhetskonfigurasjon ===
# Velger GPU (cuda) hvis tilgjengelig, ellers MPS (Apple) hvis tilgjengelig, ellers CPU.
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f" Using device: {device} ")

# === Klasse-rekkefølge (bruk A foreløpig) ===
# Viktig: denne må samsvare med rekkefølgen brukt av ImageFolder under trening.
# Hvis train.py printet ['real','ai'], bytt rekkefølge her.
CLASS_NAMES = ["ai", "real"]

# === Samme transform som i train.py ===
# Compose gjør transformasjoner i den rekkefølgen de er listet.
# 1) Resize til 224x224
# 2) ToTensor: Henter (C,H,W) med pikselverdier i [0,1]
# 3) Normalize: subtraherer mean og deler på std per kanal
# Merk: mean for grønn kanal i train.py er 0.56 (sannsynlig skrivefeil), men vi matcher det her.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),  # størrelse som brukt under trening
    transforms.ToTensor(),          # konverterer PIL -> Tensor
    transforms.Normalize([0.485, 0.56, 0.406],  # mean per kanal (R, G, B) — holder likt som train.py
                         [0.229, 0.224, 0.225])  # std per kanal (R, G, B)
])

# === Bygg modellen lik den som ble trent ===
def build_model():
    """
    Oppretter en ResNet18-instans og bytter ut fullt tilkoblet lag
    slik det var i train.py: Linear -> ReLU -> Dropout -> Softmax.
    """
    model = models.resnet18(pretrained=True)  # feature extractor (samme base)
    # Fryser feature-extractor-parametere (ikke nødvendig for inferens, men holder struktur lik treningskode)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),  # matcher train.py
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Softmax(dim=1)          # modellen gir sannsynligheter
    )
    return model

# === Last modellvekter fra fil ===
def load_model(model_path: str = "model.pth", device_override: str = None):
    """
    Laster state_dict fra model_path, plasserer modellen på valgt device og returnerer modellen.
    - model_path: sti til model.pth (state_dict) som ble lagret i train.py
    - device_override: sett "cpu" eller "cuda" om ønskelig (valgfritt)
    """
    if device_override is not None:
        dev = torch.device(device_override)
    else:
        dev = device

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fant ikke modellfil: {model_path} — sørg for at model.pth ligger i prosjektmappen.")

    model = build_model()
    # Last state_dict og håndter mapping til riktig device
    state = torch.load(model_path, map_location=dev)
    # state forventes å være state_dict fordi train.py brukte torch.save(model.state_dict(), ...)
    if isinstance(state, dict):
        # Noen ganger kan nøklene ha "module." prefiks hvis trent med DataParallel — fjern ved behov
        try:
            model.load_state_dict(state)
        except RuntimeError:
            new_state = {}
            for k, v in state.items():
                new_k = k.replace("module.", "") if k.startswith("module.") else k
                new_state[new_k] = v
            model.load_state_dict(new_state)
    else:
        # Hvis hele modellen ved et uhell var lagret, prøv å bruke direkte (ikke vanlig her)
        model = state

    model.to(dev)
    model.eval()  # viktig: sett i eval-modus for inferens
    return model

# === Forbehandle ett bilde ===
def preprocess_image(image_path: str):
    """
    Laster et bilde fra disk, konverterer til RGB, og bruker samme transform
    som ble brukt i trening. Returnerer en torch tensor med batch-dimensjon.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Bilde ikke funnet: {image_path}")

    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0)  # legger til batch-dimensjon: [1, C, H, W]
    return tensor

# === Gjør prediksjon på ett bilde ===
def predict(image_path: str, model, device_override: str = None):
    """
    Kjører modellen på gitt bilde og returnerer dict med predikert klasse,
    sannsynlighet og score per klasse.
    Eksempel retur:
        {"class": "ai", "probability": 0.923, "scores": {"ai":0.923, "real":0.077}}
    """
    dev = torch.device(device_override) if device_override else device
    model.to(dev)

    tensor = preprocess_image(image_path).to(dev)

    with torch.no_grad():
        outputs = model(tensor)           # Utdata forventes å være sannsynligheter (softmax)
        probs = outputs.squeeze(0).cpu().tolist()  # [p_class0, p_class1, ...]

    # Finn indeks for høyeste sannsynlighet
    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    result = {
        "class": CLASS_NAMES[top_idx],
        "probability": float(probs[top_idx]),
        "scores": {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
    }
    return result

# === CLI-støtte: kjør fra terminal ===
def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Predict AI vs Real image using trained ResNet18")
    parser.add_argument("image", help="Path til bildet som skal predikeres")
    parser.add_argument("--model", default="model.pth", help="Sti til model.pth (state_dict) — standard: model.pth")
    parser.add_argument("--device", default=None, help="Valgfritt: 'cpu' eller 'cuda' eller 'mps'. Standard: auto-detect")
    parser.add_argument("--json", action="store_true", help="Skriv ut JSON i stedet for menneskevennlig tekst")
    args = parser.parse_args()

    # Last modell
    model = load_model(args.model, device_override=args.device)

    # Kjør prediksjon
    res = predict(args.image, model, device_override=args.device)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"\nBilde: {args.image}")
        print(f" Predikert: {res['class']} ({res['probability']*100:.2f}%)")
        print(" Scores:")
        for k, v in res["scores"].items():
            print(f"   {k}: {v*100:.2f}%")

# Kjør kun ved direkte kjøring av scriptet
if __name__ == "__main__":
    _cli()
