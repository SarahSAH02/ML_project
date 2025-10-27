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
model_path = "model.pth" #hvor modellen lagres

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


train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform = transform ) #laster bilder fra mappen som PyTorch. Transform brukes for bildeforbehandling

subset_ratio = 0.3  # 30% av treningsdata
train_dataset.samples = random.sample(train_dataset.samples, int(len(train_dataset.samples) * subset_ratio)) #tar tilfeldig ut en mindre delmengde av dataen basert subset_ratio


val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform = transform) # laster valuderingsbildene fra data_dir/val med samme transformering
test_datasets = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform) # Laster inn testbilde fra data_dir/test med samme transformering


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True) #dataloader som grupperer treningsbilder i bacther. Shuffle = True sørger for randomisert rekkefølge
val_loader = DataLoader(val_dataset,batch_size=batch_size) # dataloader for valideringsdata. Ingen shuffle fordi vi vil ha stabile og repeterbare resultater
test_loader = DataLoader(test_datasets, batch_size=batch_size) # dataloader for testdata

class_names = train_dataset.classes #finner navnene på alle klassene basert på mappe navn. 
print(f" Classes: {class_names}") 

#Modell (ResNet18 transfer learning )

model = models.resnet18(pretrained = True) # her blir ResNet- 18 lastet. Det er et pretrained model som er trent på ImageNet.
for param in model.parameters():
    param.requires_grad = False  #fryser alle parametere i modellen. Kalles for feature extraction. 

num_fitrs = model.fc.in_features 
model.fc = nn.Sequential(
    nn.Linear(num_fitrs, 256),  # Map input til 256 nye neuroner
    nn.ReLU(), #aktiveringsfunksjon
    nn.Dropout(0.3), # forhindrer overfitting
    nn.Softmax(dim=1) # gjør output om til sannsynligheter for hver klasse
)  # bytter ut orginiale output laget med ny sekvens av lag


model = model.to(device) #flytter modellen til enten GPU eller CPU device vi bruker. Her ble det brukt CPU


criterion = nn.CrossEntropyLoss()  #definerer loss som sammenligner output med fasit labels. 

optimizer = optim.Adam(model.fc.parameters(), lr = learning_rate) # Oppretter optimizer, oppdaterer bare det nye fc laget. 

#treningsloop 
best_val_acc = 0.0  # her lagres beste valideringsnøyaktighet så langt.
train_accs, val_accs = [], [] # disse brukes for å lagre resultater for hvert epoch.

for epoch in range(num_epochs):  #kjører en treningsrunde per epoch. 
    model.train()                  #setter modellen i treningsmodus.
    running_loss, running_corrects = 0.0, 0 #initialiserer tellere.

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols = 80): #Itererer gjennom mini - batcher
        inputs, labels = inputs.to(device) , labels.to(device)  #flytter data til enten CPU eller GPU
        optimizer.zero_grad()   #nullstiller gradienter
        outputs = model(inputs)
        loss = criterion(outputs, labels) #sammenligner predisjonen med fasiten. 
        loss.backward() #gradienten regnes ut
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
    val_accs.append( val_acc.item())  #Valideringsnøyaktighet regnes ut. 

    print(f"Epoch {epoch+1}: Loss= {epoch_loss:.4f}, Train Acc = {epoch_acc}, Val Acc= {val_acc:.4f}") # printer statistikk for gjennomført treningsrunde. 

    if val_acc > best_val_acc:         #Dersom modellen presterer bedre enn før: lagres til filen model.pth. 
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


#Evaluering
test_acc = np.mean(np.array(y_true) == np.array(y_pred))
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Klassifikasjons rapport
print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - AI vs Real Images")
plt.show()

# treningskurve plotting
plt.figure()
plt.plot(range(1, num_epochs + 1), train_accs, label='Train Acc')
plt.plot(range(1, num_epochs + 1), val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()