# predict.py
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import json

# === Enhetskonfigurasjon ===
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"Using device: {device}")

# === Klasse-rekkefølge ===
CLASS_NAMES = ["ai", "real"]

# === Transform ===
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.56, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Bygg modellen ===
def build_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Softmax(dim=1)
    )
    return model

# === Last modell fra fil ===
def load_model(model_path: str = "model.pth", device_override: str = None):
    dev = torch.device(device_override) if device_override else device

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fant ikke modellfil: {model_path}")

    model = build_model()
    state = torch.load(model_path, map_location=dev)

    if isinstance(state, dict):
        try:
            model.load_state_dict(state)
        except RuntimeError:
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(new_state)
    else:
        model = state

    model.to(dev)
    model.eval()
    return model

# === Preprocess ett bilde ===
def preprocess_image(image_input):
    """
    image_input: filsti (str), BytesIO eller PIL.Image
    """
    if isinstance(image_input, str) and os.path.exists(image_input):
        image = Image.open(image_input).convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")

    tensor = TRANSFORM(image).unsqueeze(0)
    return tensor

# === Prediksjon ===
def predict(image_input, model, device_override: str = None):
    dev = torch.device(device_override) if device_override else device
    model.to(dev)

    tensor = preprocess_image(image_input).to(dev)

    with torch.no_grad():
        outputs = model(tensor)
        probs = outputs.squeeze(0).cpu().tolist()

    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    result = {
        "class": CLASS_NAMES[top_idx],
        "probability": float(probs[top_idx]),
        "scores": {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
    }
    return result
