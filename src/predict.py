import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# === Device konfigurasjon ===
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f" Using device: {device} ")

# === Samme bildetransformasjoner som i train.py ===
# Alt inni listen gjøres i rekkefølge:
# 1. Skalerer bildet
# 2. Konverterer til PyTorch tensor
# 3. Normaliserer fargekanaler
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.56, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Klassene må være i samme rekkefølge som under trening ===
class_names = ["ai", "real"]
# Hvis train.py printet: ['real', 'ai'], bytt rekkefølge her!

# === Last samme modellstruktur som ble trent ===
model = models.resnet18(pretrained=True)

# Fryser alle lag (feature extraction)
for param in model.parameters():
    param.requires_grad = False

# Bytter ut det siste laget for klassifisering (samme som i train.py)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Softmax(dim=1)  # gir sannsynligheter for hver klasse
)

# Laster inn lagrede vekter fra model.pth (beste modellen)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()  # setter modellen i evalueringsmodus (ingen trening)

# === Funksjon for å gjøre prediksjon på et bilde ===
def predict_image(img_path):
    # Åpner bildet og konverterer til RGB (3 kanaler)
    image = Image.open(img_path).convert("RGB")

    # Utfører transformasjoner og legger til batch-dimensjon (1 bilde)
    img_tensor = transform(image).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():  # skrur av gradienter for raskere utførelse
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)  # velger klassen med høyest sannsynlighet

    predicted_class = class_names[predicted.item()]
    confidence = torch.max(output).item() * 100  # konfidens i prosent

    return predicted_class, confidence


# === Kjøre skriptet fra terminal ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Bruk: python predict.py <bildefil>")
        sys.exit()

    img_path = sys.argv[1]
    label, conf = predict_image(img_path)
    print(f"Prediksjon: {label} ({conf:.2f} % sikkerhet)")
