import numpy as np
import time
import torch
from tempfile import TemporaryDirectory
import pandas as pd
import cv2 as cv
from cv2 import Mat
from tqdm import tqdm
import cv2
from lib.constants import *


def alteraImagem(image: Mat):
    image = image.astype (np.float32) / 255
    image = cv2.resize(image, (224, 224))
    
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
                    torch.save(model.state_dict(), MODEL_TO_LOAD)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model