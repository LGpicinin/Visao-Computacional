import random
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
from lib.constants import *
from lib.classes import CustomImageDataset
from lib.functions import alteraImagem, alteraLabels, train_model


random.seed (1)

# if not os.path.exists(LABEL_FILE):
print(f"Criando {LABEL_FILE}")
alteraLabels()


# fixa a semente para o gerador do número aleatório para que divisão treino/validação/teste
# seja sempre a mesma (isso é feito para que tanto no arquivo main.py quanto no metrics.py
# seja gerado a mesma divisão)
torch.manual_seed(10)

dataset = CustomImageDataset(LABEL_FILE, IMAGES_PATH, alteraImagem, torch.tensor)
train, validation, test = torch.utils.data.random_split(
                                dataset, 
                                [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT]
                            )

# Treina modelo.

# Gera a rede.
nn = torchvision.models.vit_b_32 (weights = 'ViT_B_32_Weights.IMAGENET1K_V1')

for param in nn.parameters(): # Congela tudo para o transfer learning.
    param.requires_grad = False        

# Adiciona uma camada para as 3 saídas.
nn.heads.head = torch.nn.Sequential (
        torch.nn.Linear (nn.heads.head.in_features, 3), 
        torch.nn.Softmax (dim=1)
    )

# print (nn)
nn.to (DEVICE)
train_dataloader = DataLoader(train, DATALOADER_BATCH, shuffle=True)
validation_dataloader = DataLoader(validation, DATALOADER_BATCH)

dataloaders = {
    'train': train_dataloader,
    'val': validation_dataloader
}

dataset_sizes = {
    'train': TOTAL_SAMPLES * TRAIN_SPLIT,
    'val': TOTAL_SAMPLES * VALIDATION_SPLIT
}

# Calculate class frequencies
unique_labels, counts = np.unique(dataset.img_labels['label'], return_counts=True)
class_frequencies = counts / len(dataset.img_labels)
 
# Calculate class weights as the inverse of frequencies
class_weights = 1.0 / class_frequencies
 
# Convert to PyTorch tensor
class_weights = torch.tensor(class_weights, dtype=torch.float32)

criterion = torch.nn.CrossEntropyLoss (weight=class_weights)
optimizer = torch.optim.Adam (nn.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
train_model(nn, dataloaders, dataset_sizes, criterion, optimizer, scheduler, N_EPOCH)

