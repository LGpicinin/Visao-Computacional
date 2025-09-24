import cv2 as cv
import os

WORKDIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(WORKDIR, 'images_gz2', 'images') # supondo que vamos usar o dataset do kaggle: https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images

# classificação dos registros do dataset: https://data.galaxyzoo.org/


def main ():

    for file in sorted(os.scandir(IMAGES_PATH), key=lambda x: x.name): 

        image = cv.imread (file.path, cv.IMREAD_COLOR)

        image.imshow("galaxy image", image)
    

if __name__ == '__main__':
    main ()
