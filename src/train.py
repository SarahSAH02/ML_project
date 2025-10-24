import os 
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm 

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np

import random

# konfigurer parametere
data_dir = "data/processed"  #formaterte dataene modellen skal trenes med ligger her
batch_size = 32  #i ett treningssteg skal det behandles en batch på 32 
num_epochs = 6 #antall ganger du trener over hele datasettet
learning_rate = 0.0001 #hvor raskt modellen lærer
model_path = "model.path" #hvor modellen lagres

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f" Using device: {device} ")

# Data forberedelse 

#Compose brukes til å sette sammen flere bilde- tranformasjoner
# i rekkefølge. Alt inni listen ([...]) blir gjort i den rekkefølgen man skriver det.
# På våre skjer det slik:
# 1. bildet blir resized
# 2. bildet blir konvertert til tensor
# 3. bildet blir normalisert

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # str på bildet 224*224 piksler
    transforms.ToTensor(), # konverterer bildet fra et vanlig bildeformat til en PyTorch tensor.
    transforms.Normalize([0.485, 0.56, 0.406], # gjennomsnitt (mean) for hver kanal
                         [0.229, 0.224, 0.225]) # standardavvik (std) for hver kanal
]) #normaliserer pikselverdiene for hver fargekanal (R,G,B)


train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform = transform )
subset_ratio = 0.3  # 30% av treningsdata
train_dataset.samples = random.sample(train_dataset.samples, int(len(train_dataset.samples) * subset_ratio))


val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform = transform)
test_datasets = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True)
val_loader = DataLoader(val_dataset,batch_size=batch_size)
test_loader = DataLoader(test_datasets, batch_size=batch_size)

class_names = train_dataset.classes
print(f" Classes: {class_names}")

#Modell (ResNet50 transfer learning)

model = models.resnet18(pretrained = True)
for param in model.parameters():
    param.requires_grad = False

num_fitrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_fitrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Softmax(dim=1)
)


model = model.to(device)

#tap og optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr = learning_rate)

#treningsloop
best_val_acc = 0.0
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols = 80):
        inputs, labels = inputs.to(device) , labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    train_accs.append(epoch_acc.item())

    model.eval()
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device) , labels.to(device)
            outputs = model(inputs)
            _ , preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
    
    val_acc = val_corrects.double() / len(val_dataset)
    val_accs.append( val_acc.item())

    print(f"Epoch {epoch+1}: Loss= {epoch_loss:.4f}, Train Acc = {epoch_acc}, Val Acc= {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f" Lagret ny beste modell med val_acc = {val_acc:.4f}")

print("Trening ferdig!")

# laste beste modell og test

model.load_state_dict(torch.load(model_path))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())


# === Evaluering ===
test_acc = np.mean(np.array(y_true) == np.array(y_pred))
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Klassifikasjonsrapport
print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - AI vs Real Images")
plt.show()

# Treningskurve
plt.figure()
plt.plot(range(1, num_epochs + 1), train_accs, label='Train Acc')
plt.plot(range(1, num_epochs + 1), val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()