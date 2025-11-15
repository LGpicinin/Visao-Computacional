import os
import torch

DATA_PATH = './data'
IMAGES_PATH = f'{DATA_PATH}/images'

CLASS_FILE = os.path.join(DATA_PATH, "gz2_hart16.csv")
MAPPING_FILE = os.path.join(DATA_PATH, "gz2_filename_mapping.csv")

LABEL_FILE = os.path.join(DATA_PATH, "labels.csv")

TOTAL_SAMPLES = 5000

WIDTH = 224

MODEL_TO_LOAD = os.path.join(DATA_PATH, "saved.pth")
TRAIN_SPLIT = 0.5
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.3
N_EPOCH = 1 # Com poucas epocas, já funciona.
LEARNING_RATE = 0.001
DATALOADER_BATCH = 2000

DEVICE = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')

ESC_KEY = 27
