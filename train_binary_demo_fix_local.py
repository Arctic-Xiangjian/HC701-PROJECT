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
import collections

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset

from sklearn.metrics import f1_score, accuracy_score

sys.path.append('/home/hong/hc701/HC701-PROJECT')
from VAL import test

import torch.backends.cudnn as cudnn

import timm
from timm.models import create_model

import copy

from hc701fed.dataset.dataset_list_transform import (
    MESSIDOR_binary_pairs_train,
    MESSIDOR_binary_pairs_test,
    MESSIDOR_binary_Etienne_train,
    MESSIDOR_binary_Etienne_test,
    MESSIDOR_binary_Brest_train,
    MESSIDOR_binary_Brest_test
)

centerlized_train = ConcatDataset([MESSIDOR_binary_pairs_train, MESSIDOR_binary_Etienne_train, MESSIDOR_binary_Brest_train])
centerlized_test = ConcatDataset([MESSIDOR_binary_pairs_test, MESSIDOR_binary_Etienne_test, MESSIDOR_binary_Brest_train])


batch_size = 8


def test(model_, test_loader, device):
    model_test = copy.deepcopy(model_)
    model_test.to(device)
    model_test.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_test(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    return y_true, y_pred

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Data loader
train_loader_pairs = DataLoader(MESSIDOR_binary_pairs_train, batch_size=batch_size, shuffle=True)
train_loader_Etienne = DataLoader(MESSIDOR_binary_Etienne_train, batch_size=batch_size, shuffle=True)
train_loader_Brest = DataLoader(MESSIDOR_binary_Brest_train, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(centerlized_train, batch_size=batch_size, shuffle=True)

test_loader_pairs = DataLoader(MESSIDOR_binary_pairs_test, batch_size=1, shuffle=False)
test_loader_Etienne = DataLoader(MESSIDOR_binary_Etienne_test, batch_size=1, shuffle=False)
test_loader_Brest = DataLoader(MESSIDOR_binary_Brest_test, batch_size=1, shuffle=False)
test_loader = DataLoader(centerlized_test, batch_size=1, shuffle=False)

train_list = [train_loader_pairs, train_loader_Etienne, train_loader_Brest]
test_list = [test_loader_pairs, test_loader_Etienne, test_loader_Brest]


for noise_scale in [0.1,0.2,0.3,0.5,1]:
    for seed in [42,43,44]:
        print('seed: {}, noise_scale: {}'.format(seed, noise_scale))
        cudnn.deterministic = True
        cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        
        batch_size = 8

        # use resnet18 as the base model
        model = create_model('vgg16', pretrained=False, num_classes=2)

        # load the fix model
        model_name = 'vgg16'
        model_path = '/home/hong/hc701/random_init/{}.pth'.format(model_name)
        model.load_state_dict(torch.load(model_path))

        mean = torch.tensor(0, dtype=torch.float)
        std = torch.tensor(0.0948*noise_scale, dtype=torch.float)
        lr=0.01
        clip_value=30

        def local_train(trains_loader_item, num_updates,model_init,client_id,rounds,lr=lr):
            model_init.to(device)
            __model=copy.deepcopy(model_init)
            __model.to(device)
            # Local train
            loss = torch.nn.CrossEntropyLoss()
            optimizer=torch.optim.SGD(__model.parameters(),lr=lr)
            __model.train()
            update_count = 0
            for i, (X, y) in enumerate(trains_loader_item):
                update_count += 1
                X, y = X.to(device), y.to(device)
                # Compute prediction and loss
                pred = __model(X)
                _loss = loss(pred,y)
                # Backpropagation
                _loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if update_count >= num_updates:
                    break
            return __model,model_init
        
        def get_update(model_previous: torch.nn.modules,model_t: torch.nn.modules):
            updates=torch.tensor([]).to(device).reshape(1,-1)
            for p_t,p_pre in zip(model_t.parameters(),model_previous.parameters()):
                updates=torch.cat((updates,torch.flatten(p_t-p_pre).reshape(1,-1)),1)
            return updates
        
        def apply_dp(update_tensor,total_round,clip_threshold=clip_value,learn_rate=lr,dp_std=std,dp_mean=mean):
            # clip gradient
            updates_norm=torch.linalg.vector_norm(update_tensor)
            clip_threshold=learn_rate*clip_threshold
            update_tensor_clip=update_tensor/torch.max(torch.tensor([1]).to(device),updates_norm/clip_threshold)
            # add noise
            update_tensor_clip+=torch.normal(mean=dp_mean, std=learn_rate*dp_std*clip_threshold*total_round,size=update_tensor_clip.shape).to('cuda:2')
            return update_tensor_clip
        
        def update_model(dp_update_tensor: torch.Tensor(), model_pre: torch.nn.modules,model_t: torch.nn.modules):
            dp_update_tensor=torch.flatten(dp_update_tensor)  # Flatten update make it to 1D
            # Save model dict
            pre_dict_model=model_pre.state_dict()
            t_dict_model=model_t.state_dict()
            #update by the DP guarantee delta
            for name, param in model_t.named_parameters():
                length_paramter=int((torch.flatten(param).shape)[0]) # Get the length of of paramter
                delta=dp_update_tensor[:length_paramter].reshape(param.shape)  # Delta is the parameter grad times the lr
                t_dict_model[name]=pre_dict_model[name]+delta  # update model
                dp_update_tensor=dp_update_tensor[length_paramter:] # remove used
            model_t.load_state_dict(t_dict_model) # update model
            return model_t
        
        def local_dp_train(trains_loader_item, num_updates,model_init,client_id,rounds):
            model_t,model_pre=local_train(trains_loader_item=trains_loader_item, num_updates=num_updates,model_init=model_init,client_id=client_id,rounds=rounds)
            model_update=get_update(model_pre,model_t)
            # print(model_update)
            dp_update_tensor=apply_dp(update_tensor=model_update,total_round=rounds)
            # print(dp_update_tensor)
            model_t_dp=update_model(dp_update_tensor=dp_update_tensor,model_pre=model_pre,model_t=model_t)
            return model_t_dp
        
        def local_step(training_dataloaders_list,num_updates,model_init,rounds):
            MODEL_LIST=[model_init for i in range(3)]
            for i,j in zip(training_dataloaders_list,range(3)):
                MODEL_LIST[j]=local_dp_train(trains_loader_item=i,num_updates=num_updates,model_init=model_init,client_id=j,rounds=rounds)
            return MODEL_LIST
        
        def aggregation_model(models_list):
            _models_list=copy.deepcopy(models_list)
            fed_state_dict=collections.OrderedDict()
            weight_keys=models_list[0].state_dict().keys()
            for key in weight_keys:
                key_avg=0
                for _model in _models_list:
                    key_avg=key_avg+_model.state_dict()[key]*1/3
                fed_state_dict[key]=key_avg
            for _model in _models_list:
                _model.load_state_dict(fed_state_dict)
            return _models_list
        
        for com_round in [1,2,5,10,20,100,500,1000,2000]:
            global_models=[copy.deepcopy(model) for i in range(3)]
            for rounds in tqdm(range(com_round)):
                models=local_step(train_list,50,global_models[0],com_round)
                global_models=aggregation_model(models)
            y_true,y_pred=test(global_models[0],test_loader,device)
            print('Round: {} Accuracy: {}'.format(com_round,accuracy_score(y_true,y_pred)))
            print('Round:',com_round,'f1_score:',f1_score(y_true,y_pred,average='macro'))
            print('Round:',com_round,'average_Acc_f1:',(accuracy_score(y_true,y_pred)+f1_score(y_true,y_pred,average='macro'))/2)
            print('\n')
    print('-------------------------------------------------------------------------------------')