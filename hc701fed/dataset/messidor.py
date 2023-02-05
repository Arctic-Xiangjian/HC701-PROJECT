import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import utils

data_dir_options = {
    'messidor2': '/l/users/xiangjian.hou/preprocessed/messidor2',
    'messidor_pairs' : '/l/users/xiangjian.hou/preprocessed/messidor/messidor_pairs',
    'messidor_Etienne' : '/l/users/xiangjian.hou/preprocessed/messidor/messidor_Etienne',
    'messidor_Brest-without_dilation' : '/l/users/xiangjian.hou/preprocessed/messidor/messidor_Brest-without_dilation',
}

class MESSIDOR(Dataset):
    def __init__(self, data_dir, transform=None, mode='train',train_val_split=0.25):
        self.data_dir = data_dir
        self.transform = transforms.ToTensor() if transform is None else transform
        self.mode = mode
        self.train_val_split = train_val_split

        self.data = []
        self.labels = []

        if self.mode == 'train' or self.mode == 'val':
            data_dir = os.path.join(self.data_dir, 'train')
            for label in os.listdir(data_dir):
                for img in os.listdir(os.path.join(data_dir, label)):
                    self.data.append(os.path.join(data_dir, label, img))
                    self.labels.append(int(label))
            self.data, self.val_data, self.labels, self.val_labels = train_test_split(self.data, self.labels, test_size=self.train_val_split, random_state=42)
        elif self.mode == 'test':
            data_dir = os.path.join(self.data_dir, 'test')
            for img in os.listdir(data_dir):
                self.data.append(os.path.join(data_dir, img))
                self.labels.append(0)
        else:
            raise ValueError('mode should be train, val or test')


    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        elif self.mode == 'val':
            return len(self.val_data)
        elif self.mode == 'test':
            return len(self.data)
        else:
            raise ValueError('mode should be train, val or test')
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.transform(self.data[idx]), self.labels[idx]
        elif self.mode == 'val':
            return self.transform(self.val_data[idx]), self.val_labels[idx]
        else:
            return self.transform(self.data[idx]), self.labels[idx]

    