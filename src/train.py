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

# konfigurer parametere
data_dir = "data/processed"  #formaterte dataene modellen skal trenes med ligger her
batch_size = 32  #i ett treningssteg skal det behandles en batch på 32 
num_epochs = 10 #
learning_rate = 0.0001 
model_path = "model.path"

