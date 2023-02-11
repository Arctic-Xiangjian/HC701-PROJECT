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
import json

import torch
from torch.utils.data import Dataset, DataLoader

from VAL import test

from hc701fed.dataset.WeightedConcatDataset import WeightedConcatDataset
from hc701fed.dataset.dataset_list_transform import (
    APTOS_train,
    EyePACS_train,
    MESSIDOR_2_train,
    MESSIDOR_pairs_train,
    MESSIDOR_Etienne_train,
    MESSIDOR_Brest_train,
    APTOS_Val,
    EyePACS_Val,
    MESSIDOR_2_Val,
    MESSIDOR_pairs_Val,
    MESSIDOR_Etienne_Val,
    MESSIDOR_Brest_Val,
)

from hc701fed.model.baseline import (
    Baseline
)


def main(backbone,
         lr, batch_size, epochs, device, optimizer,
         dataset,seed, use_wandb, 
         wandb_project, wandb_entity,
         save_model, checkpoint_path,
         num_classes=5
         ):
    
    if save_model:
        if checkpoint_path == "none":
            raise ValueError("checkpoint_path is None")
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
    
    # set seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    # load dataset
    if dataset == "centerlized":
        Centerlized_train = WeightedConcatDataset([APTOS_train, EyePACS_train, MESSIDOR_2_train, MESSIDOR_pairs_train, MESSIDOR_Etienne_train,MESSIDOR_Brest_train])
        Centerlized_Val = WeightedConcatDataset([APTOS_Val, EyePACS_Val, MESSIDOR_2_Val, MESSIDOR_pairs_Val, MESSIDOR_Etienne_Val, MESSIDOR_Brest_Val])
        train_dataset = DataLoader(Centerlized_train, batch_size=batch_size, shuffle=True)
        val_dataset = DataLoader(Centerlized_Val, batch_size=batch_size, shuffle=False)
        LOSS = torch.nn.CrossEntropyLoss(weight=Centerlized_train.calculate_weights())
    elif dataset == "messidor":
        MESSIDOR_Centerlized_train = WeightedConcatDataset([MESSIDOR_pairs_train, MESSIDOR_Etienne_train,MESSIDOR_Brest_train])
        MESSIDOR_Centerlized_Val = WeightedConcatDataset([MESSIDOR_pairs_Val, MESSIDOR_Etienne_Val, MESSIDOR_Brest_Val])
        train_dataset = DataLoader(MESSIDOR_Centerlized_train, batch_size=batch_size, shuffle=True)
        val_dataset = DataLoader(MESSIDOR_Centerlized_Val, batch_size=batch_size, shuffle=False)
        num_classes = 4
        LOSS = torch.nn.CrossEntropyLoss(weight=MESSIDOR_Centerlized_train.calculate_weights())
    elif dataset == "aptos":
        train_dataset = DataLoader(APTOS_train, batch_size=batch_size, shuffle=True)
        val_dataset = DataLoader(APTOS_Val, batch_size=batch_size, shuffle=False)
        LOSS = torch.nn.CrossEntropyLoss(weight=APTOS_train.calculate_weights())
    elif dataset == "eyepacs":
        train_dataset = DataLoader(EyePACS_train, batch_size=batch_size, shuffle=True)
        val_dataset = DataLoader(EyePACS_Val, batch_size=batch_size, shuffle=False)
        LOSS = torch.nn.CrossEntropyLoss(weight=EyePACS_train.calculate_weights())
    elif dataset == "messidor2":
        train_dataset = DataLoader(MESSIDOR_2_train, batch_size=batch_size, shuffle=True)
        val_dataset = DataLoader(MESSIDOR_2_Val, batch_size=batch_size, shuffle=False)
        LOSS = torch.nn.CrossEntropyLoss(weight=MESSIDOR_2_train.calculate_weights())
    elif dataset == "messidor_pairs":
        train_dataset = DataLoader(MESSIDOR_pairs_train, batch_size=batch_size, shuffle=True)
        val_dataset = DataLoader(MESSIDOR_pairs_Val, batch_size=batch_size, shuffle=False)
        num_classes = 4
        LOSS = torch.nn.CrossEntropyLoss(weight=MESSIDOR_pairs_train.calculate_weights())
    elif dataset == "messidor_etienne":
        train_dataset = DataLoader(MESSIDOR_Etienne_train, batch_size=batch_size, shuffle=True)
        val_dataset = DataLoader(MESSIDOR_Etienne_Val, batch_size=batch_size, shuffle=False)
        num_classes = 4
        LOSS = torch.nn.CrossEntropyLoss(weight=MESSIDOR_Etienne_train.calculate_weights())
    elif dataset == "messidor_brest":
        train_dataset = DataLoader(MESSIDOR_Brest_train, batch_size=batch_size, shuffle=True)
        val_dataset = DataLoader(MESSIDOR_Brest_Val, batch_size=batch_size, shuffle=False)
        num_classes = 4
        LOSS = torch.nn.CrossEntropyLoss(weight=MESSIDOR_Brest_train.calculate_weights())
    else:
        raise NotImplementedError

    # load model
    model = Baseline(backbone=backbone, num_classes=num_classes)
    model.to(device)
    
    # print something to prove so far so good
    print(f'You are using {backbone} backbone, lr is {lr}, batch_size is {batch_size}, epochs is {epochs}, device is {device}, optimizer is {optimizer}, dataset is {dataset}, seed is {seed}, checkpoint_path is {checkpoint_path}, num_classes is {num_classes}')


    if use_wandb:
        run = wandb.init(project=wandb_project, entity=wandb_entity, name=dataset+'_'+backbone+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'), job_type="training",reinit=True)

    # optimizer str to class
    optimizer = eval(optimizer)
    optimizer = optimizer(model.parameters(), lr=lr)

    # train
    model_begin_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_f1 = 0
    best_acc = 0
    count_no_improve = 0
    for epoch in range(epochs):
        model.train()
        model.to(device)
        epoch_loss = 0
        for i, (x, y) in enumerate(tqdm(train_dataset)):
            x = x.to(device,torch.float32)
            y = y.to(device,torch.long)
            optimizer.zero_grad()
            pred = model(x)
            loss = LOSS(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # wandb log Step: epoch
        epoch_loss = epoch_loss / len(train_dataset) * batch_size
        # validation
        acc, f1 = test(model, device, val_dataset)
        # wandb log Step: epoch
        if use_wandb:
            # Horizontal axis: epoch
            wandb.log({"val_acc": acc, "val_f1": f1,"train_loss": epoch_loss})
        # save model every time after validation get better f1_score
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            count_no_improve = 0
            if save_model:
                if not os.path.exists(os.path.join(checkpoint_path, dataset+'_'+backbone+'_'+str(seed))):
                    os.mkdir(os.path.join(checkpoint_path, dataset+'_'+backbone+'_'+str(seed)))
                save_path_meta = os.path.join(checkpoint_path, dataset+'_'+backbone+'_'+str(seed))
                if not os.path.exists(os.path.join(save_path_meta, model_begin_time)):
                    os.mkdir(os.path.join(save_path_meta, model_begin_time))
                save_path = os.path.join(save_path_meta, model_begin_time)
                torch.save(model.state_dict(), os.path.join(save_path, f"{dataset}_{backbone}_{epoch}_{model_begin_time}.pth"))
                torch.save(model.state_dict(), os.path.join(save_path, f"{dataset}_{backbone}_best.pth"))
                if epoch == epochs-1:
                    torch.save(model.state_dict(), os.path.join(checkpoint_path, dataset+'_'+backbone+'_'+str(seed), f"{dataset}_{backbone}_last.pth"))
        # if the f1_score is not getting better for 5 epochs, stop training
        if f1 < best_f1:
            count_no_improve += 1
            if count_no_improve >= 7:
                break

    if use_wandb:
        run.finish()

    train_set_acc , train_set_f1 = test(model, device, train_dataset)
    # Save the config of the model as a json file and the best f1_score of validation set and the f1_score of train set with last epoch to see if the model is overfitting
    if save_model:
        optimizer = str(optimizer).split(" ")[0]
        with open(os.path.join(save_path, f"{dataset}_{backbone}_{model_begin_time}.json"), "w") as f:
            json.dump({"backbone": backbone, "lr": lr, "batch_size": batch_size, "epochs": epoch, "device": device, "optimizer": optimizer, "dataset": dataset, "seed": seed, "best_acc": best_acc, "best_f1": best_f1, "train_set_f1": train_set_f1, "train_set_acc": train_set_acc}, f)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--optimizer", type=str, default='torch.optim.AdamW')
    parser.add_argument("--dataset", type=str, default="centerlized", help='centerlized or not', choices=['centerlized', "messidor","aptos" , "eyepacs" , "messidor2", "messidor_pairs","messidor_etienne","messidor_brest"])
    parser.add_argument("--seed", type=int, default=42)
    # wandb true or false
    parser.add_argument("--use_wandb", action='store_true', help='use wandb or not')
    parser.add_argument("--wandb_project", type=str, default="HC701-PROJECT")
    parser.add_argument("--wandb_entity", type=str, default="arcticfox")
    # save model    
    parser.add_argument("--save_model", action='store_true', help='save model or not')
    parser.add_argument("--checkpoint_path", type=str, default='none')
    args = parser.parse_args()
    main(**vars(args))