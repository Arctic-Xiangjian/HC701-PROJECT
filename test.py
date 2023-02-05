import sys
import numpy as np
import pandas as pd
import os
import cv2
import wandb
from datetime import datetime
from tqdm import tqdm
import argparse
import random
import copy

import torch
from torch.utils.data import Dataset, DataLoader


from hc701fed.dataset.dataset_list import (
    Centerlized_Val,
    MESSIDOR_Centerlized_Val
)

from hc701fed.model.baseline import (
    Baseline,
    BACKBONES
)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def test(test_model, device, val_dataset):
    model = copy.deepcopy(test_model)
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, label in tqdm(val_dataset):
            data = data.to(device,torch.float32)
            label = label.to(device,torch.long)
            output = model(data)
            y_true.append(label.cpu().numpy())
            y_pred.append(output.cpu().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_pred = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred,average='macro')
        auc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
    return acc, f1, auc