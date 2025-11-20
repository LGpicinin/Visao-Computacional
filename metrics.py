import numpy as np
import random
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.constants import *
from lib.functions import showImages, alteraImagem, alteraLabels
from lib.classes import CustomImageDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


SHOW_IMAGES = False

alteraLabels()
random.seed (1)

# fixa a semente para o gerador do número aleatório para que divisão treino/validação/teste
# seja sempre a mesma (isso é feito para que tanto no arquivo main.py quanto no metrics.py
# seja gerado a mesma divisão)
torch.manual_seed(10)


dataset = CustomImageDataset(LABEL_FILE, IMAGES_PATH, alteraImagem, torch.tensor)
train, validation, test = torch.utils.data.random_split(
                                dataset, 
                                [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT]
                            )

model, _ = MODEL_AND_WEIGHT
nn = model ()
if isinstance(nn, torchvision.models.efficientnet.EfficientNet):
    nn.classifier = torch.nn.Sequential (
            torch.nn.Linear (nn.classifier[-1].in_features, 2), 
            torch.nn.Softmax (dim=1)
        )
else:
# Adiciona uma camada para as 3 saídas.
    nn.heads.head = torch.nn.Sequential (
            torch.nn.Linear (nn.heads.head.in_features, 2), 
            torch.nn.Softmax (dim=1)
        )

nn.to (DEVICE)
nn.load_state_dict (torch.load (MODEL_TO_LOAD, weights_only=True, map_location=DEVICE))
nn.eval()
if SHOW_IMAGES:
    showImages(nn)
    
# build confusion matrix on the whole dataset and print metrics
dataloader = DataLoader(test, batch_size=DATALOADER_BATCH, shuffle=False)
num_classes = 3
conf_mat = np.zeros((num_classes, num_classes), dtype=int)
all_preds = []
all_labels = []
nn.eval()
y_test = []
predictions = []

with torch.no_grad():
    for inputs, labels in tqdm(dataloader):

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE).long().view(-1)


        outputs = nn(inputs)
        preds = torch.argmax(outputs, dim=1)

        preds_cpu = preds.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        for t, p in zip(labels_cpu, preds_cpu):
            y_test.append(t)
            predictions.append(p)
            conf_mat[t, p] += 1
        all_preds.append(preds_cpu)
        all_labels.append(labels_cpu)

cm = confusion_matrix(y_test, predictions)
ConfusionMatrixDisplay(cm).plot() 
plt.show()

if len(all_preds) > 0:
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
else:
    all_preds = np.array([])
    all_labels = np.array([])


# print("Confusion matrix (rows=true, cols=predicted):")
# print(conf_mat)
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