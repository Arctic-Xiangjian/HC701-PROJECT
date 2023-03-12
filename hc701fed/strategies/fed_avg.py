import time
from typing import List

import torch
from torch.utils.data import DataLoader, ConcatDataset

import os
import json
import numpy as np
from tqdm import tqdm
import random
import copy
import wandb
import argparse
import collections
from datetime import datetime
import sys

sys.path.append(os.path.join(os.getcwd(), 'HC701-PROJECT'))
sys.path.append('../../')

from hc701fed.dataset.dataset_list_transform import (
    APTOS_train,
    EyePACS_train,
    MESSIDOR_2_train,
    MESSIDOR_pairs_train,
    MESSIDOR_Etienne_train,
    MESSIDOR_Brest_train,
)

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

from hc701fed.dataset.WeightedConcatDataset import WeightedConcatDataset

from hc701fed.model.baseline import (
    Baseline
)
from VAL import test

def local_step(model, train_dataloader, optimizer, LOSS,lr, device, local_steps=100):
    model_to_train = copy.deepcopy(model)
    optimizer = optimizer(model_to_train.parameters() , lr = lr)
    model_to_train.train()
    model_to_train.to(device)
    ls = 0
    while ls < local_steps:
        for batch_idx, (data, target) in tqdm(enumerate(train_dataloader)):
            data, target = data.to(device), target.to(device,torch.long)
            optimizer.zero_grad()
            output = model_to_train(data)
            loss = LOSS(output, target)
            loss.backward()
            optimizer.step()
            ls += 1
            if ls >= local_steps:
                break
    return model_to_train

def fed_avg(backbone,lr, batch_size, device, optimizer,
            seed, use_wandb, 
            wandb_project, wandb_entity,
            save_model, checkpoint_path,
            use_scheduler,
            # FedAvg parameters
            num_local_epochs, num_comm_rounds,data_set_mode,
):
    
    # set seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    if save_model and checkpoint_path== 'none':
        raise ValueError("checkpoint_path should be specified when save_model is True")

    if data_set_mode == "datasets":
        MESSIDOR_Center_train = ConcatDataset([MESSIDOR_pairs_train, MESSIDOR_Etienne_train, MESSIDOR_Brest_train])
        MESSIDOR_Center_val = ConcatDataset([MESSIDOR_pairs_Val, MESSIDOR_Etienne_Val, MESSIDOR_Brest_Val])
        MESSIDOR_Center_test = ConcatDataset([MESSIDOR_pairs_Test, MESSIDOR_Etienne_Test, MESSIDOR_Brest_Test])
        train_dataset_list = [APTOS_train, EyePACS_train, MESSIDOR_2_train, MESSIDOR_Center_train]
        dataset_list_str = ["APTOS", "EyePACS", "MESSIDOR_2", "MESSIDOR"]
        val_dataset_list = [APTOS_Val, EyePACS_Val, MESSIDOR_2_Val, MESSIDOR_Center_val]
        test_dataset_list = [APTOS_Test, EyePACS_Test, MESSIDOR_2_Test, MESSIDOR_Center_test]
        Centrilized_val = ConcatDataset([APTOS_Val, EyePACS_Val, MESSIDOR_2_Val, MESSIDOR_pairs_Val, MESSIDOR_Etienne_Val, MESSIDOR_Brest_Val])
        Centrilized_test = ConcatDataset([APTOS_Test, EyePACS_Test, MESSIDOR_2_Test, MESSIDOR_pairs_Test, MESSIDOR_Etienne_Test, MESSIDOR_Brest_Test])
    elif data_set_mode == "hosptials":
        train_dataset_list = [MESSIDOR_pairs_train, MESSIDOR_Etienne_train, MESSIDOR_Brest_train]
        val_dataset_list = [MESSIDOR_pairs_Val, MESSIDOR_Etienne_Val, MESSIDOR_Brest_Val]
        dataset_list_str = ["MESSIDOR_pairs", "MESSIDOR_Etienne", "MESSIDOR_Brest"]
        test_dataset_list = [MESSIDOR_pairs_Test, MESSIDOR_Etienne_Test, MESSIDOR_Brest_Test]
        Centrilized_val = ConcatDataset([MESSIDOR_pairs_Val, MESSIDOR_Etienne_Val, MESSIDOR_Brest_Val])
        Centrilized_test = ConcatDataset([MESSIDOR_pairs_Test, MESSIDOR_Etienne_Test, MESSIDOR_Brest_Test])
    else:
        raise ValueError("data_set_mode should be either datasets or hosptials")
    

    Centrilized_val_dataloader = DataLoader(dataset=Centrilized_val, batch_size=256, shuffle=False, num_workers=4)
    Centrilized_test_dataloader = DataLoader(dataset=Centrilized_test, batch_size=256, shuffle=False, num_workers=4)

    # Initialize the wandb
    if use_wandb:
        run = wandb.init(project=wandb_project, entity=wandb_entity, name=data_set_mode+'_'+backbone+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'), job_type="training",reinit=True)

    # Initialize the model
    model = Baseline(backbone=backbone,num_classes=5)
    model_keys = model.state_dict().keys()
    # Initialize the optimizer
    optimizer = eval(optimizer)
    # Initialize the loss function
    LOSS = torch.nn.CrossEntropyLoss()
    # Initialize the scheduler

    # Initialize the global model
    model_global = copy.deepcopy(model)


    # Initialize the list of datasets

    # Initialize the list of dataloaders
    train_dataloader_list = [DataLoader(dataset=train_dataset_list[i], batch_size=batch_size, shuffle=True, num_workers=4) for i in range(len(train_dataset_list))]

    total_train_iters = sum([len(train_dataloader_list[i]) for i in range(len(train_dataset_list))])

    # val_dataloader_list = [DataLoader(dataset=val_dataset_list[i], batch_size=batch_size, shuffle=False, num_workers=4) for i in range(len(val_dataset_list))]

    test_dataloader_list = [DataLoader(dataset=test_dataset_list[i], batch_size=batch_size, shuffle=False, num_workers=4) for i in range(len(test_dataset_list))]
    
    # Start the communication rounds
    best_f1 = 0
    model_begin_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    non_improving_rounds = 0
    lr_with_decay = copy.deepcopy(lr)
    if use_scheduler:
        model_for_scheduler = copy.deepcopy(model)
        optimizer_scheduler = copy.deepcopy(optimizer(model_for_scheduler.parameters(), lr=lr))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_scheduler, T_max=10,eta_min=0.00001)
    for comm_round in range(num_comm_rounds):
        # Local training
        models_list_new = []
        for i, train_dataloader_iter in enumerate(train_dataloader_list):
            model_new = local_step(model=model_global, train_dataloader=train_dataloader_iter, optimizer=optimizer, LOSS=LOSS,lr=lr_with_decay, device=device)
            # Try to print the learning rate to see whether the scheduler works
            # print(optimizer.param_groups[0]['lr'])
            models_list_new.append(copy.deepcopy(model_new))
        # FedAvg
        global_state_dict = collections.OrderedDict()
        # Delete the gradients of all the models
        for key in model_keys:
            key_avg = 0
            for model, train_dataloader_iter in zip(models_list_new, train_dataloader_list):
                key_avg += model.state_dict()[key]*len(train_dataloader_iter)/total_train_iters
            global_state_dict[key] = key_avg
        model_global.load_state_dict(global_state_dict)
        # make lr_with_decay decay
        if use_scheduler:
            scheduler.step()
            lr_with_decay = optimizer_scheduler.param_groups[0]['lr']


        # Validation
        val_model = model_global
        val_acc,val_f1 = test(val_model,device,Centrilized_val_dataloader)

        if use_wandb:
            wandb.log({"val_acc": val_acc, "val_f1": val_f1})
        
        if save_model:
            if val_f1 > best_f1:
                non_improving_rounds = 0
                best_f1 = val_f1
                save_path_meta = os.path.join(checkpoint_path, 'fed_avg_'+data_set_mode+'_'+str(seed))
                if not os.path.exists(save_path_meta):
                    os.makedirs(save_path_meta)
                save_model_path = os.path.join(save_path_meta, model_begin_time)
                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)
                torch.save(val_model.state_dict(), os.path.join(save_model_path, 'fed_avg'+'_'+data_set_mode+'_'+backbone+'_'+str(comm_round)+'_model.pt'))
                torch.save(val_model.state_dict(), os.path.join(save_model_path, 'fed_avg'+'_'+data_set_mode+'_'+backbone+'_'+'best_model.pt'))
                print('best model saved')
            else:
                non_improving_rounds += 1
                if non_improving_rounds >= 80:
                    break
    if use_wandb:
        run.finish()
    # Test
    # load the best model
    val_model.load_state_dict(torch.load(os.path.join(save_model_path, 'fed_avg'+'_'+data_set_mode+'_'+backbone+'_'+'best_model.pt')))
    test_result = {}
    Centrilized_test_acc,Centrilized_test_f1 = test(val_model,device,Centrilized_test_dataloader)
    print('Centrilized_test_acc: ', Centrilized_test_acc, 'Centrilized_test_f1: ', Centrilized_test_f1)
    test_result['Centrilized_test_acc'] = Centrilized_test_acc
    test_result['Centrilized_test_f1'] = Centrilized_test_f1
    for dataset_name, dataset_for_test in zip(dataset_list_str, test_dataloader_list):
        test_acc,test_f1 = test(val_model,device,dataset_for_test)
        test_result[dataset_name+'_test_acc'] = test_acc
        test_result[dataset_name+'_test_f1'] = test_f1
        print(dataset_name+'_test_acc: ', test_acc, dataset_name+'_test_f1: ', test_f1)
    # try to sava some config for clarity
    if save_model:
        with open(os.path.join(save_model_path, 'config.json'), 'w') as f:
            json.dump(vars(args), f)
        with open(os.path.join(save_model_path, 'test_result.json'), 'w') as f:
            json.dump(test_result, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--optimizer', type=str, default='torch.optim.AdamW')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--use_wandb", action='store_true', help='use wandb or not')
    parser.add_argument('--wandb_project', type=str, default='FedAvg_hc701')
    parser.add_argument("--wandb_entity", type=str, default="arcticfox")
    parser.add_argument("--save_model", action='store_true', help='save model or not')
    parser.add_argument('--checkpoint_path', type=str, default='none')
    parser.add_argument('--use_scheduler', action='store_true', help='use scheduler or not')
    # FedAvg parameters
    parser.add_argument('--num_local_epochs', type=int, default=1)
    parser.add_argument('--num_comm_rounds', type=int, default=500)
    parser.add_argument('--data_set_mode', type=str, default='datasets',choices=['datasets','hosptials'])
    args = parser.parse_args()
    fed_avg(**vars(args))

        
        
