import os
import torch
import torchvision

DATA_PATH = './data'
IMAGES_PATH = f'{DATA_PATH}/images'

CLASS_FILE = os.path.join(DATA_PATH, "gz2_hart16.csv")
MAPPING_FILE = os.path.join(DATA_PATH, "gz2_filename_mapping.csv")

LABEL_FILE = os.path.join(DATA_PATH, "labels.csv")

TOTAL_SAMPLES = 5000 # 239695 é o max

WIDTH = 224

MODEL_TO_LOAD = os.path.join(DATA_PATH, "saved.pth")
TRAIN_SPLIT = 0.5
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.3
N_EPOCH = 1 # Com poucas epocas, já funciona.
LEARNING_RATE = 0.001
DATALOADER_BATCH = 2000
# MODEL_AND_WEIGHT = torchvision.models.efficientnet_v2_s, 'EfficientNet_V2_S_Weights.IMAGENET1K_V1'
MODEL_AND_WEIGHT = torchvision.models.vit_b_32, 'ViT_B_32_Weights.IMAGENET1K_V1'

DEVICE = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')

ESC_KEY = 27
