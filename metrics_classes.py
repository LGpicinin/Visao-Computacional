import numpy as np

from lib.constants import *
from lib.functions import showImages, alteraImagem, alteraLabels, controiDataframePesos
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd


def rodaMetricas(file_name: str):
    df_pesos = pd.read_csv(os.path.join(DATA_PATH, f"{file_name}.csv"), sep=',')

    df_smooth = df_pesos[df_pesos['smooth'] > df_pesos['disk']]
    df_disk = df_pesos[df_pesos['smooth'] < df_pesos['disk']]


    infos = f"""
    {file_name}:
        SMOOTH GALAXIES:
            MEAN SMOOTH: {df_smooth['smooth'].mean()}
            MEAN DISK: {df_smooth['disk'].mean()}
            MEAN STAR/ARTIFACT: {df_smooth['star'].mean()}
            STD SMOOTH: {df_smooth['smooth'].std()}
        DISK GALAXIES:
            MEAN DISK: {df_disk['disk'].mean()}
            MEAN SMOOTH: {df_disk['smooth'].mean()}
            MEAN STAR/ARTIFACT: {df_disk['star'].mean()}
            STD DISK: {df_disk['disk'].std()}
    """

    print(infos)

    pesos_infos = open(os.path.join(DATA_PATH, f"{file_name}.txt"), 'w')
    pesos_infos.write(infos)
    pesos_infos.close()

rodaMetricas("pesos_right_en")
rodaMetricas("pesos_wrong_en")