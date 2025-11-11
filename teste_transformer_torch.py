#===============================================================================
# EXEMPLO: RETÂNGULO X CÍRCULO X TRIÂNGULO COM AMOSTRAS SINTÉTICAS
#===============================================================================
# Teste usando a ViT nativa do PyTorch pré-treinada no ImageNet, com transfer
# learning ajustando somente a última camada.

import random
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import cv2 as cv
from cv2 import Mat
import matplotlib.pyplot as plt
import os
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

TOTAL_SAMPLES = 239695

WIDTH = 224

TRAIN = True
MODEL_TO_LOAD = 'saved.pth'
TRAIN_SPLIT = 0.5
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.3
N_EPOCH = 1 # Com poucas epocas, já funciona.
LEARNING_RATE = 0.001
DATALOADER_BATCH = 2000

SHOW_IMAGES = False

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


def showImages():
    key = 'a'
    while key != ESC_KEY:
        cv2.destroyAllWindows()
        data = pd.read_csv(LABEL_FILE)
        sample = data.sample(1)
        img_name = sample['image_name'].values[0]
        label = sample['label'].values[0]
        
        if not os.path.exists(f'{IMAGES_PATH}/{img_name}.jpg'):
            print(f'{IMAGES_PATH}/{img_name}.jpg')
            continue
        
        image = cv.imread(f'{IMAGES_PATH}/{img_name}.jpg', cv.IMREAD_COLOR)
        # tensor_img = np.empty((1, 3, WIDTH, WIDTH), np.float32)
        # tensor_img[0] = alteraImagem(image).unsqueeze(0)
        tensor_img = alteraImagem(image).unsqueeze(0)

        with torch.no_grad():
            result = nn (tensor_img)

        print('%.4f %.4f %.4f' % (result[0][0], result[0][1], result[0][2]))

        result = np.argmax (result[0])      
        labels = ['elliptical', 'spiral', 'artifact_or_star']  
        print(f'Predict: {labels[result]}, real: {labels[label]}')

        cv2.imshow (f'{img_name}', image)
        key = cv2.waitKey ()

    cv2.destroyAllWindows ()


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


#===============================================================================
# Script.

random.seed (1)

dataset = CustomImageDataset(LABEL_FILE, IMAGES_PATH, alteraImagem, torch.tensor)
train, validation, test = torch.utils.data.random_split(dataset, [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT])

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

    criterion = torch.nn.CrossEntropyLoss ()
    optimizer = torch.optim.Adam (nn.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    train_model(nn, dataloaders, dataset_sizes, criterion, optimizer, scheduler, N_EPOCH)

else:
    nn = torchvision.models.vit_b_32 ()
    # Adiciona uma camada para as 3 saídas.
    nn.heads.head = torch.nn.Sequential (torch.nn.Linear (nn.heads.head.in_features, 3), torch.nn.Softmax (dim=1))
    nn.to (DEVICE)
    nn.load_state_dict (torch.load (MODEL_TO_LOAD, weights_only=True, map_location=DEVICE))
    nn.eval()

    alteraLabels()

    if SHOW_IMAGES:
        showImages()
        
    # build confusion matrix on the whole dataset and print metrics

    dataloader = DataLoader(test, batch_size=DATALOADER_BATCH, shuffle=False)

    num_classes = 3
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    all_preds = []
    all_labels = []

    nn.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long().view(-1)
            outputs = nn(inputs)
            preds = torch.argmax(outputs, dim=1)

            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()

            for t, p in zip(labels_cpu, preds_cpu):
                conf_mat[t, p] += 1

            all_preds.append(preds_cpu)
            all_labels.append(labels_cpu)

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
    else:
        all_preds = np.array([])
        all_labels = np.array([])

    print("Confusion matrix (rows=true, cols=predicted):")
    print(conf_mat)

    total = conf_mat.sum()
    accuracy = conf_mat.trace() / total if total > 0 else 0.0
    with open('all.txt', 'w') as f:
        print(f"Overall accuracy: {accuracy:.4f}")
        f.write(f"Overall accuracy: {accuracy:.4f}\n")

        # per-class precision, recall, f1
        for i in range(num_classes):
            tp = conf_mat[i, i]
            fp = conf_mat[:, i].sum() - tp
            fn = conf_mat[i, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            print(f"Class {i}: precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")
            f.write(f"Class {i}: precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}\n")
