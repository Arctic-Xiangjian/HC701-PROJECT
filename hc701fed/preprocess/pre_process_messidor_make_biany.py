import torch

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm

DATA_PATH_META = '/home/hong/hc701/preprocessed/messidor_biany'

for i in os.listdir(DATA_PATH_META):
    for j in os.listdir(os.path.join(DATA_PATH_META, i)):
        for k in os.listdir(os.path.join(DATA_PATH_META, i, j)):
            data = np.load(os.path.join(DATA_PATH_META, i, j, k), allow_pickle=True).item()
            if data['label'] == 0:
                pass
            else:
                data['label'] = 1
            np.save(os.path.join(DATA_PATH_META, i, j, k), data)