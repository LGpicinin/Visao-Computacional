#===============================================================================
# EXEMPLO: RETÂNGULO X CÍRCULO X TRIÂNGULO COM AMOSTRAS SINTÉTICAS
#===============================================================================
# Teste usando a ViT nativa do PyTorch pré-treinada no ImageNet, com transfer
# learning ajustando somente a última camada.

from cProfile import label
from email.mime import image
import random
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2 as cv
from cv2 import Mat
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset

#===============================================================================
# CONFIG
#===============================================================================

DATA_PATH = './data'
IMAGES_PATH = f'{DATA_PATH}/images'

CLASS_FILE = os.path.join(DATA_PATH, "gz2_hart16.csv")
MAPPING_FILE = os.path.join(DATA_PATH, "gz2_filename_mapping.csv")

LABEL_FILE = os.path.join(DATA_PATH, "labels.csv")

TOTAL_SAMPLES = 10000

WIDTH = 500

TRAIN = True
MODEL_TO_LOAD = 'saved.pth'
N_TRAINING = 2048
N_VALIDATION = 512
N_EPOCH = 1 # Com poucas epocas, já funciona.
LEARNING_RATE = 0.001
BATCH_SIZE = 1

DEVICE = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')

ESC_KEY = 27


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 'image_name'], '.jpg')
        image = cv.imread(img_path, cv.IMREAD_COLOR)
        label = self.img_labels.iloc[idx, "simple_class"]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

def alteraImagem(image: Mat):
    image = image.astype (np.float32) / 255
    image = cv2.resize(image, (224, 224))
    return image


def alteraLabels():
    '''
        Função que salva csv com nome da imagem e respectiva label
    '''
    classes = pd.read_csv(CLASS_FILE, usecols=['dr7objid', 'gz2_class'])
    classes = classes.sample(TOTAL_SAMPLES)

    classes['simple_class'] = (
        classes['gz2_class']
            .str
            .replace('^E.*$', 'elliptical', regex=True)
            .replace('^S.*$', 'spiral', regex=True)
            .replace('^A$', 'artifact_or_star', regex=True)        
    )
    
    classes['label'] = (
        classes['simple_class']
            .str
            .replace('elliptical', '0')
            .replace('spiral', '1')
            .replace('artifact_or_star', '2')
            .astype(np.int8)
    )

    filename_mapping = pd.read_csv(MAPPING_FILE)

    classes = classes.merge(filename_mapping, left_on='dr7objid', right_on='objid')
    
    classes = classes.rename(columns={'asset_id': 'image_name'})

    cols = ["image_name", "simple_class"]
    classes = classes[cols]

    classes.to_csv(LABEL_FILE, index=False)



#===============================================================================
# TREINO
#===============================================================================

def trainNetwork (nn, train_x, train_y, validation_x, validation_y):
    '''Gera exemplos aleatórios e treina uma CNN.'''

    criterion = torch.nn.CrossEntropyLoss ()
    optimizer = torch.optim.Adam (nn.parameters(), lr=LEARNING_RATE)

    # Converte para tensores.
    train_x = torch.tensor (train_x.transpose((0,3,1,2)))
    train_y = torch.tensor (train_y)
    validation_x = torch.tensor (validation_x.transpose((0,3,1,2)), dtype=torch.float32)
    validation_y = torch.tensor (validation_y)

    # Normalização.
    # Não é estritamente necessário para treinar do zero, esta média e desvio-
    # padrão são importantes para usar o modelo pré-treinado no imagenet.
    norm = torchvision.transforms.Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_x = norm (train_x)
    validation_x = norm (validation_x)

    train_dataset = TensorDataset (train_x, train_y)
    val_dataset = TensorDataset (validation_x, validation_y)
    train_loader = DataLoader (train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader (val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []


    # Para cada época...
    for epoch in range(N_EPOCH):
        nn.train ()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad ()
            outputs = nn (inputs)
            loss = criterion (outputs, labels)
            loss.backward ()
            optimizer.step ()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / total)
        train_accs.append(correct / total)

        # Validação
        nn.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = nn (inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / total)
        val_accs.append(correct / total)

        print(f"Epoch {epoch+1}/{N_EPOCH}, "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

        # Salva melhor modelo
        if val_accs[-1] == max(val_accs):
            torch.save(nn.state_dict(), "saved.pth")

    # Plots
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    plt.savefig("training.png")
    plt.clf()

    plt.plot(train_accs, label='train_acc')
    plt.plot(val_accs, label='val_acc')
    plt.legend()
    plt.savefig("training_acc.png")


#===============================================================================
# Script.

random.seed (1)

# Treina ou carrega o modelo.
if TRAIN:
    # Gera a rede.
    nn = torchvision.models.vit_b_32 (weights = 'ViT_B_32_Weights.IMAGENET1K_V1')
    for param in nn.parameters(): # Congela tudo para o transfer learning.
        param.requires_grad = False        
    # Adiciona uma camada para as 3 saídas.
    nn.heads.head = torch.nn.Sequential (torch.nn.Linear (nn.heads.head.in_features, 3), torch.nn.Softmax (dim=1))
    # print (nn)
    nn.to (DEVICE)
    
    alteraLabels()

    dataset = CustomImageDataset(LABEL_FILE, IMAGES_PATH, alteraImagem, torchvision.transforms.ToTensor())
    
    train, validation = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train, 500, shuffle=True)
    validation_dataloader = DataLoader(validation, 500)
    # train_x, validation_x, train_y, validation_y = train_test_split(dataset, test_size=0.2, random_state=42)

    
    # train_x = np.array(train_x)
    # validation_x = np.array(validation_x)
    # train_y = np.array(train_y)
    # validation_y = np.array(validation_y)
    
    trainNetwork (nn, train_x, train_y, validation_x, validation_y)
# else:
#     nn = torchvision.models.vit_b_32 ()
#     # Adiciona uma camada para as 3 saídas.
#     nn.heads.head = torch.nn.Sequential (torch.nn.Linear (nn.heads.head.in_features, 3), torch.nn.Softmax (dim=1))
#     print (nn)
#     nn.to (DEVICE)
#     nn.load_state_dict (torch.load (MODEL_TO_LOAD, weights_only=True, map_location=DEVICE))
#     nn.eval()

#     # Testa. Gera imagens de teste uma a uma.
#     img = np.empty ((1, WIDTH, WIDTH, 3), np.float32)
#     key = 'a'
#     while key != ESC_KEY:
#         generateImage (img [0])
#         tensor_img = torch.tensor (img.transpose ((0,3,1,2))).to(DEVICE)

#         with torch.no_grad():
#             result = nn (tensor_img)

#         print('%.4f %.4f %.4f' % (result[0][0], result[0][1], result[0][2]))

#         shape = np.argmax (result[0])        
#         if shape == 0:
#             print ('Circulo')
#         elif shape == 1:
#             print ('Retangulo')
#         else:
#             print ('Triângulo')

#         cv2.imshow ('oi', img [0])
#         key = cv2.waitKey ()

#     cv2.destroyAllWindows ()
