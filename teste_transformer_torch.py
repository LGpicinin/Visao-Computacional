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
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2 as cv
from cv2 import Mat
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
from tempfile import TemporaryDirectory
from tqdm import tqdm
import time

#===============================================================================
# CONFIG
#===============================================================================

DATA_PATH = './data'
IMAGES_PATH = f'{DATA_PATH}/images'

CLASS_FILE = os.path.join(DATA_PATH, "gz2_hart16.csv")
MAPPING_FILE = os.path.join(DATA_PATH, "gz2_filename_mapping.csv")

LABEL_FILE = os.path.join(DATA_PATH, "labels.csv")

TOTAL_SAMPLES = 50000

WIDTH = 500

TRAIN = True
MODEL_TO_LOAD = 'saved.pth'
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.2
N_EPOCH = 1 # Com poucas epocas, já funciona.
LEARNING_RATE = 0.001
BATCH_SIZE = 1
DATALOADER_BATCH = 2000

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
        img_path = os.path.join(self.img_dir, str(self.img_labels.loc[idx, 'image_name']))
        img_path = f'{img_path}.jpg'
        image = cv.imread(img_path, cv.IMREAD_COLOR)
        label = self.img_labels.loc[idx, "label"]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

def alteraImagem(image: Mat):
    image = image.astype (np.float32) / 255
    image = cv2.resize(image, (224, 224))
    
    return torch.tensor(image.transpose((2, 0, 1)))


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
    
    classes['label'] = pd.to_numeric(
        classes['simple_class']
            .str
            .replace('elliptical', '0')
            .replace('spiral', '1')
            .replace('artifact_or_star', '2')
            
    )

    filename_mapping = pd.read_csv(MAPPING_FILE)

    classes = classes.merge(filename_mapping, left_on='dr7objid', right_on='objid')
    
    classes = classes.rename(columns={'asset_id': 'image_name'})

    cols = ["image_name", "label"]
    classes = classes[cols]
    classes = classes[[
        os.path.isfile(
            f'{os.path.join(IMAGES_PATH, str(img_name))}.jpg' 
        ) for img_name in classes['image_name']
    ]]

    classes.to_csv(LABEL_FILE, index=False)



#===============================================================================
# TREINO
#===============================================================================
# código tirado daqui (acom algumas adaptações): https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model


# def trainNetwork (nn, train_x, train_y, validation_x, validation_y):
#     '''Gera exemplos aleatórios e treina uma CNN.'''

#     criterion = torch.nn.CrossEntropyLoss ()
#     optimizer = torch.optim.Adam (nn.parameters(), lr=LEARNING_RATE)

#     # Converte para tensores.
#     train_x = torch.tensor (train_x.transpose((0,3,1,2)))
#     train_y = torch.tensor (train_y)
#     validation_x = torch.tensor (validation_x.transpose((0,3,1,2)), dtype=torch.float32)
#     validation_y = torch.tensor (validation_y)

#     # Normalização.
#     # Não é estritamente necessário para treinar do zero, esta média e desvio-
#     # padrão são importantes para usar o modelo pré-treinado no imagenet.
#     norm = torchvision.transforms.Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     train_x = norm (train_x)
#     validation_x = norm (validation_x)

#     train_dataset = TensorDataset (train_x, train_y)
#     val_dataset = TensorDataset (validation_x, validation_y)
#     train_loader = DataLoader (train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader (val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#     train_losses, val_losses, train_accs, val_accs = [], [], [], []


#     # Para cada época...
#     for epoch in range(N_EPOCH):
#         nn.train ()
#         running_loss, correct, total = 0.0, 0, 0

#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

#             optimizer.zero_grad ()
#             outputs = nn (inputs)
#             loss = criterion (outputs, labels)
#             loss.backward ()
#             optimizer.step ()

#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         train_losses.append(running_loss / total)
#         train_accs.append(correct / total)

#         # Validação
#         nn.eval()
#         val_loss, correct, total = 0.0, 0, 0
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#                 outputs = nn (inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item() * inputs.size(0)
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         val_losses.append(val_loss / total)
#         val_accs.append(correct / total)

#         print(f"Epoch {epoch+1}/{N_EPOCH}, "
#               f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, "
#               f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

#         # Salva melhor modelo
#         if val_accs[-1] == max(val_accs):
#             torch.save(nn.state_dict(), "saved.pth")

#     # Plots
#     plt.plot(train_losses, label='train_loss')
#     plt.plot(val_losses, label='val_loss')
#     plt.legend()
#     plt.savefig("training.png")
#     plt.clf()

#     plt.plot(train_accs, label='train_acc')
#     plt.plot(val_accs, label='val_acc')
#     plt.legend()
#     plt.savefig("training_acc.png")


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

    dataset = CustomImageDataset(LABEL_FILE, IMAGES_PATH, alteraImagem, torch.tensor)
    
    train, validation = torch.utils.data.random_split(dataset, [TRAIN_SPLIT, VALIDATION_SPLIT])
    train_dataloader = DataLoader(train, DATALOADER_BATCH, shuffle=True)
    validation_dataloader = DataLoader(validation, DATALOADER_BATCH)
    # train_x, validation_x, train_y, validation_y = train_test_split(dataset, test_size=0.2, random_state=42)

    
    # train_x = np.array(train_x)
    # validation_x = np.array(validation_x)
    # train_y = np.array(train_y)
    # validation_y = np.array(validation_y)
    dataloaders = {
        'train': train_dataloader,
        'val': validation_dataloader
    }
    
    dataset_sizes = {
        'train': TOTAL_SAMPLES * TRAIN_SPLIT,
        'val': TOTAL_SAMPLES * VALIDATION_SPLIT
    }

    criterion = torch.nn.CrossEntropyLoss ()
    optimizer = torch.optim.Adam (nn.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    train_model(nn, dataloaders, dataset_sizes, criterion, optimizer, scheduler, N_EPOCH)
    
    # trainNetwork (nn, train_x, train_y, validation_x, validation_y)
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
