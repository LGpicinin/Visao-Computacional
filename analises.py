import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import os

DATA_PATH = './data'
IMAGES_PATH = f'{DATA_PATH}/images'

classes = pd.read_csv(f"{DATA_PATH}/gz2_hart16.csv", usecols=['dr7objid', 'gz2_class'])

classes['simple_class'] = (
    classes['gz2_class']
        .str
        .replace('^E.*$', 'elliptical', regex=True)
        .replace('^S.*$', 'spiral', regex=True)
        .replace('^A$', 'artifact_or_star', regex=True)        
)

classes['simple_class'].hist(bins=3)
plt.show()

filename_mapping = pd.read_csv(f'{DATA_PATH}/gz2_filename_mapping.csv')

a = ''
while a not in (27, 113):
    sample = classes.sample(1)
    obj_id = sample['dr7objid'].values[0]
    asset_id = filename_mapping[filename_mapping['objid'] == obj_id]['asset_id'].values[0]
    
    if not os.path.exists(f'{IMAGES_PATH}/{asset_id}.jpg'):
        print(f'{IMAGES_PATH}/{asset_id}.jpg')
        continue
    
    image = cv.imread(f'{IMAGES_PATH}/{asset_id}.jpg', cv.IMREAD_COLOR)

    cv.imshow(f'Image type: {sample["simple_class"].values[0]}', image)
    a = cv.waitKey()
    cv.destroyAllWindows()
    