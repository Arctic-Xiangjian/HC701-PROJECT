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

import torch
from torch.utils.data import Dataset, DataLoader


from hc701fed.dataset.dataset_list import (
    Centerlized_train,
    MESSIDOR_Centerlized_train
)

from hc701fed.model.baseline import (
    Baseline,
    BACKBONES
)

LOSS = torch.nn.CrossEntropyLoss()

def main(backbone,
         lr, batch_size, epochs, device, 
         dataset,seed, use_wandb, 
         wandb_project,wandb_entity, wandb_run_name, wandb_tags, wandb_notes,
         num_classes=5):
    
    # set seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    # load dataset
    if dataset == "centerlized":
        train_dataset = DataLoader(Centerlized_train, batch_size=batch_size, shuffle=True)
    elif dataset == "messidor":
        train_dataset = DataLoader(MESSIDOR_Centerlized_train, batch_size=batch_size, shuffle=True)
        num_classes = 4
    else:
        raise NotImplementedError

    # load model
    model = Baseline(backbone=backbone, num_classes=num_classes)
    model.to(device)

    if use_wandb:
        run = wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name+'_'+dataset+'_'+{datetime.now().strftime('%Y%m%d_%H%M%S')}, tags=wandb_tags, notes=wandb_notes,reinit=True)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    for epoch in range(epochs):
        model.train()
        model.to(device)
        train_loss = 0
        train_acc = 0
        for i, (x, y) in enumerate(tqdm(train_dataset)):
            x = x.to(device,torch.float32)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = LOSS(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (pred.argmax(dim=1) == y).sum().item()
        train_loss /= len(train_dataset)
        train_acc /= len(train_dataset.dataset)
        if use_wandb:
            wandb.log({"train_loss": train_loss, "train_acc": train_acc})
        # save model when train acc is the best and last epoch
        # best_acc = 0
        # if train_acc > best_acc:
        #     best_acc = train_acc
        #     if not os.path.exists(model_name):
        #         os.mkdir(model_name)
        #     torch.save(model.state_dict(), os.path.join(model_name, f"{backbone}_{dataset}_{model_name}_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"))
        # if epoch == epochs - 1:
        #     torch.save(model.state_dict(), os.path.join(model_name, f"{backbone}_{dataset}_{model_name}_last.pth"))
    if wandb:
        run.finish()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="densenet121")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="messidor")
    parser.add_argument("--seed", type=int, default=42)
    # wandb true or false
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="HC701-PROJECT")
    parser.add_argument("--wandb_entity", type=str, default="hc701")
    parser.add_argument("--wandb_run_name", type=str, default="baseline")
    parser.add_argument("--wandb_tags", type=str, default="baseline")
    parser.add_argument("--wandb_notes", type=str, default="baseline")
    args = parser.parse_args()
    main(**vars(args))