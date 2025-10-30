import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# === Device configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device} ")

# === Same transforms as in train.py ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize([0.485, 0.56, 0.406],  
                         [0.229, 0.224, 0.225])
])

# === Class names must match training ===
class_names = ["ai", "real"]  
# ⚠ NOTE: If train order was ["real", "ai"]
# then swap order above. Check print from train.py output.

# === Build the same ResNet model structure ===
model = models.resnet18(pretrained=True)

# Freeze all feature extractor layers
for param in model.parameters():
    param.requires_grad = False

# Replace final fully connected layer — MUST match train.py
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Softmax(dim=1)
)

# Load trained weights into model
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# ================= Prediction Function ================= #
def predict_image(img_path):
    """Loads an image → applies transform → model prediction → returns class name"""

    # Load image using Pillow
    image = Image.open(img_path).convert("RGB")

    # Apply same preprocessing as training
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension [1,3,224,224]
    img_tensor = img_tensor.to(device)

    # Disable gradient tracking
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)  # index of highest probability

    predicted_class = class_names[predicted.item()]
    confidence = torch.max(output).item() * 100

    return predicted_class, confidence


# Run prediction from terminal
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit()

    img_path = sys.argv[1]
    label, conf = predict_image(img_path)
    print(f"Prediction: {label} ({conf:.2f} %)")
