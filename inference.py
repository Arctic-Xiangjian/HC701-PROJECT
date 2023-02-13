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
import json
import csv

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset


from sklearn.metrics import accuracy_score, f1_score

from hc701fed.dataset.val_dataset_list import (
    APTOS_Val,
    EyePACS_Val,
    MESSIDOR_2_Val,
    MESSIDOR_pairs_Val,
    MESSIDOR_Etienne_Val,
    MESSIDOR_Brest_Val,
)
from hc701fed.dataset.val_dataset_list import (
    APTOS_Test,
    EyePACS_Test,
    MESSIDOR_2_Test,
    MESSIDOR_pairs_Test,
    MESSIDOR_Etienne_Test,
    MESSIDOR_Brest_Test,
)
from hc701fed.model.baseline import Baseline

def main(backbone,model_path,dataset,trained_dataset,mode,device):
    if model_path == "none":
        raise NotImplementedError("Please specify the model path")

    if mode == 'val':
        if dataset == "centerlized":
            Centerlized_Val = ConcatDataset([APTOS_Val, EyePACS_Val, MESSIDOR_2_Val, MESSIDOR_pairs_Val, MESSIDOR_Etienne_Val, MESSIDOR_Brest_Val])
            val_dataset = DataLoader(Centerlized_Val, batch_size = 256, shuffle=False)
        elif dataset == "messidor":
            MESSIDOR_Centerlized_Val = ConcatDataset([MESSIDOR_pairs_Val, MESSIDOR_Etienne_Val, MESSIDOR_Brest_Val])
            val_dataset = DataLoader(MESSIDOR_Centerlized_Val, batch_size = 256, shuffle=False)
        elif dataset == "aptos":
            val_dataset = DataLoader(APTOS_Val, batch_size = 256, shuffle=False)
        elif dataset == "eyepacs":
            val_dataset = DataLoader(EyePACS_Val, batch_size = 256, shuffle=False)
        elif dataset == "messidor2":
            val_dataset = DataLoader(MESSIDOR_2_Val, batch_size = 256, shuffle=False)
        elif dataset == "messidor_pairs":
            val_dataset = DataLoader(MESSIDOR_pairs_Val, batch_size = 256, shuffle=False)
        elif dataset == "messidor_etienne":
            val_dataset = DataLoader(MESSIDOR_Etienne_Val, batch_size = 256, shuffle=False)
        elif dataset == "messidor_brest":
            val_dataset = DataLoader(MESSIDOR_Brest_Val, batch_size = 256, shuffle=False)
        else:
            raise NotImplementedError
    elif mode == 'test':
        if dataset == "centerlized":
            Centerlized_Test = ConcatDataset([APTOS_Test, EyePACS_Test, MESSIDOR_2_Test, MESSIDOR_pairs_Test, MESSIDOR_Etienne_Test, MESSIDOR_Brest_Test])
            val_dataset = DataLoader(Centerlized_Test, batch_size = 256, shuffle=False)
        elif dataset == "messidor":
            MESSIDOR_Centerlized_Test = ConcatDataset([MESSIDOR_pairs_Test, MESSIDOR_Etienne_Test, MESSIDOR_Brest_Test])
            val_dataset = DataLoader(MESSIDOR_Centerlized_Test, batch_size = 256, shuffle=False)
        elif dataset == "aptos":
            val_dataset = DataLoader(APTOS_Test, batch_size = 256, shuffle=False)
        elif dataset == "eyepacs":
            val_dataset = DataLoader(EyePACS_Test, batch_size = 256, shuffle=False)
        elif dataset == "messidor2":
            val_dataset = DataLoader(MESSIDOR_2_Test, batch_size = 256, shuffle=False)
        elif dataset == "messidor_pairs":
            val_dataset = DataLoader(MESSIDOR_pairs_Test, batch_size = 256, shuffle=False)
        elif dataset == "messidor_etienne":
            val_dataset = DataLoader(MESSIDOR_Etienne_Test, batch_size = 256, shuffle=False)
        elif dataset == "messidor_brest":
            val_dataset = DataLoader(MESSIDOR_Brest_Test, batch_size = 256, shuffle=False)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    if trained_dataset == 'messidor' or trained_dataset == 'messidor_pairs' or trained_dataset == 'messidor_etienne' or trained_dataset == 'messidor_brest':
        num_classes = 4
    else:
        num_classes = 5

    model_load_path = os.path.join(model_path, f"{trained_dataset}_{backbone}_best.pth")
    model = Baseline(backbone = backbone, num_classes = num_classes, pretrained = False)
    model.load_state_dict(torch.load(model_load_path))
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    y_pred_prob = []

    # print inference setting
    print(f"Dataset: {dataset}, Backbone: {backbone}, Mode: {mode}, trained_dataset: {trained_dataset}, num_classes: {num_classes}, model_load_path: {model_load_path}")
    if mode == 'val':
        for i, (x, y) in enumerate(tqdm(val_dataset)):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                y_pred_prob.append(model(x).cpu().numpy())
                y_pred.append(np.argmax(model(x).cpu().numpy(), axis=1))
                y_true.append(y.cpu().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        accuracy, f1 = accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')

        print(f"Accuracy: {accuracy}, F1: {f1}")

        # save the result to a json file
        with open(os.path.join(model_path, f"{trained_dataset}_{dataset}_{backbone}_result.json"), "w") as f:
            json.dump({"test_dataset" : dataset, "trained_dataset" : trained_dataset, "accuracy" : accuracy, "f1" : f1}, f)
    elif mode == 'test' and (dataset == 'aptos' or dataset == 'centerlized' or dataset == 'eyepacs'):
        # we don't have the ground truth for test dataset, save the prediction to a csv file
        with open(os.path.join(model_path, f"{trained_dataset}_{dataset}_{backbone}_result.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id_code", "diagnosis"])
            for i, (x, fake_label, image_name) in enumerate(tqdm(val_dataset)):
                x = x.to(device)
                with torch.no_grad():
                    y_pred_prob = model(x).cpu().numpy()
                    y_pred = np.argmax(y_pred_prob, axis=1)
                    for j in range(len(y_pred)):
                        writer.writerow([image_name[j], y_pred[j]])
    else:
        # for other test dataset, we have the ground truth, save the result to a json file
        for i, (x, y, image_name) in enumerate(tqdm(val_dataset)):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                y_pred_prob.append(model(x).cpu().numpy())
                y_pred.append(np.argmax(model(x).cpu().numpy(), axis=1))
                y_true.append(y.cpu().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_pred_prob = np.concatenate(y_pred_prob)
        y_pred_prob = torch.softmax(torch.from_numpy(y_pred_prob), dim=1).numpy()
        accuracy, f1= accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')
        print(f"Accuracy: {accuracy}, F1: {f1}")

        # save the result to a json file
        with open(os.path.join(model_path, f"{trained_dataset}_{dataset}_{backbone}_test_set_result.json"), "w") as f:
            json.dump({"test_dataset" : dataset, "trained_dataset" : trained_dataset, "accuracy" : accuracy, "f1" : f1}, f)
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--model_path", type=str, default="none")
    parser.add_argument("--dataset", type=str, default="centerlized")
    parser.add_argument("--trained_dataset", type=str, default="centerlized")
    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(**vars(args))
