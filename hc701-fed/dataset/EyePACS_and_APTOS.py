import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import utils

data_dir_options = {
    'EyePACS': '/l/users/xiangjian.hou/preprocessed/eyepacs',
    'APTOS': '/l/users/xiangjian.hou/preprocessed/aptos',
}

class Eye_APTOS(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.transform = transforms.ToTensor() if self.transform is None else self.transform
        self.train = train

        if self.train:
            self.data_path = os.path.join(self.data_dir, 'train')
            self.data = []
            self.labels = []
            for i in os.listdir(self.data_path):
                data = np.load(os.path.join(self.data_path, i), allow_pickle=True).item()
                # Image to 0-1
                image_data = data['image'] / data['image'].max()
                self.data.append(image_data)
                self.labels.append(data['label'])
        else:
            self.data_path = os.path.join(self.data_dir, 'test')
            self.data = []
            for i in os.listdir(self.data_path):
                data = np.load(os.path.join(self.data_path, i), allow_pickle=True).item()
                image_data = data['image'] / data['image'].max()
                self.data.append(image_data)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.train:
            return self.transform(self.data[idx]), self.labels[idx]
        else:
            return self.transform(self.data[idx]), 0
