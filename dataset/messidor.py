import os
import numpy as np
import pandas as pd

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
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.transform = transforms.ToTensor() if self.transform is None else self.transform
        self.train = train

        if self.train:
            self.data_path = os.path.join(self.data_dir, 'train')
        else:
            self.data_path = os.path.join(self.data_dir, 'test')

        self.data = []
        self.labels = []
        for i in os.listdir(self.data_path):
            data = np.load(os.path.join(self.data_path, i), allow_pickle=True).item()
            image_data = data['image']/ data['image'].max()
            self.data.append(image_data)
            self.labels.append(data['label'])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        image = self.transform(image)
        return image, label
    