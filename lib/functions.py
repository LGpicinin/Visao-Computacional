import numpy as np
import time
import torch
from tempfile import TemporaryDirectory
import pandas as pd
import cv2 as cv
from cv2 import Mat
from tqdm import tqdm
import cv2
import json
from lib.constants import *
from matplotlib import pyplot as plt


def alteraImagem(image: Mat):
    image = image.astype (np.float32) / 255
    image = cv2.resize(image, (WIDTH, WIDTH))
    
    return torch.tensor(image.transpose((2, 0, 1)))


def showImages(nn):
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
    classes = classes.sample(TOTAL_SAMPLES, random_state=2)

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

    classes = classes[classes['label'] != 2]

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

def controiDataframePesos() -> pd.DataFrame:
    '''
        Função que retorna dataframe com nome da imagem e pesos de cada classe
    '''
    cols = []
    classes = pd.read_csv(CLASS_FILE)
    classes = classes.sample(TOTAL_SAMPLES, random_state=2)
    for col in classes.columns:
        if "t01_smooth_or_features" in col and "debiased" in col:
            cl = "disk" if "disk" in col else ("star" if "star" in col else "smooth")
            classes[cl] = classes[col]
            cols.append(cl)


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

    cols.append("label")

    classes = classes[classes['label'] != 2]

    filename_mapping = pd.read_csv(MAPPING_FILE)

    classes = classes.merge(filename_mapping, left_on='dr7objid', right_on='objid')
    
    classes = classes.rename(columns={'asset_id': 'image_name'})

    cols.append("image_name")
    classes = classes[cols]
    classes = classes[[
        os.path.isfile(
            f'{os.path.join(IMAGES_PATH, str(img_name))}.jpg' 
        ) for img_name in classes['image_name']
    ]]

    return classes

    


# código tirado daqui (acom algumas adaptações): https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler):
    since = time.time()
    
    try:
        with open('training_configs.json', 'r') as file:
            training_configs = json.loads(file.read())
    except:
        training_configs = {
            'current_epoch': 0,
            'num_epochs': N_EPOCH,
            'best_acc': 0.0,
            'losses': {
                'train': [],
                'val': []
            },
            'accs': {
                'train': [],
                'val': []
            }
        }

    last_model_params_path = os.path.join(DATA_PATH, 'last_model_params.pt')
    best_model_params_path = os.path.join(DATA_PATH, 'best_model_params.pt')
    if os.path.exists(last_model_params_path):
        model.load_state_dict(torch.load(last_model_params_path, weights_only=True))
    else:
        torch.save(model.state_dict(), best_model_params_path)

    for epoch in range(training_configs['current_epoch'], training_configs['num_epochs']):
        print(f'Epoch {epoch}/{training_configs["num_epochs"] - 1}')
        print('-' * 10)
        
        training_configs['current_epoch'] = epoch
        torch.save(model.state_dict(), last_model_params_path)
        with open('training_configs.json', 'w') as file:
            file.write(json.dumps(training_configs))

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
            
            training_configs['losses'][phase].append(float(epoch_loss))
            training_configs['accs'][phase].append(float(epoch_acc))

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > training_configs['best_acc']:
                training_configs['best_acc'] = float(epoch_acc)
                torch.save(model.state_dict(), best_model_params_path)
                torch.save(model.state_dict(), MODEL_TO_LOAD)
                
        plt.plot(training_configs['losses']['train'], label='train_loss')
        plt.plot(training_configs['losses']['val'], label='val_loss')
        plt.legend()
        plt.savefig("training.png")
        plt.clf()

        plt.plot(training_configs['accs']['train'], label='train_acc')
        plt.plot(training_configs['accs']['val'], label='val_acc')
        plt.legend()
        plt.savefig("training_acc.png")
        plt.clf()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {training_configs["best_acc"]:4f}')

    torch.save(model.state_dict(), last_model_params_path)
    with open('training_configs.json', 'w') as file:
        file.write(json.dumps(training_configs))
    
    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
        
    return model