import os
import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import utils

class MESSIDOR_binary(Dataset):
    def __init__(self, data_dir, transform_=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5145, 0.2434, 0.0807],std=[0.2870, 0.1415, 0.0531])
        ]) if transform_ is None else transform_
        self.mode = mode
        self.data = []
        self.labels = []

        if self.mode == 'train':
            self.data_path = os.path.join(self.data_dir, 'train')
            for i in os.listdir(self.data_path):
                    data = np.load(os.path.join(self.data_path, i), allow_pickle=True).item()
                    # Image to 0-1
                    image_data = data['image']
                    self.data.append(image_data)
                    self.labels.append(data['label'])
        elif self.mode == 'test':
            self.data_id = []
            self.data_path = os.path.join(self.data_dir, 'test')
            for i in os.listdir(self.data_path):
                data = np.load(os.path.join(self.data_path, i), allow_pickle=True).item()
                # Image to 0-1
                image_data = data['image']
                self.data.append(image_data)
                self.labels.append(np.array(data['label'], dtype=np.int64))
                self.data_id.append(i[:-4])
        else:
            raise ValueError('mode should be train, val or test')


    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        elif self.mode == 'test':
            return len(self.data)
        else:
            raise ValueError('mode should be train, val or test')
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.transform(self.data[idx]), self.labels[idx]
        else:
            return self.transform(self.data[idx]), self.labels[idx]
        
    def calculate_weights(self):
        if self.mode == 'train':
            labels = self.labels
        else:
            raise ValueError('mode should be train or val')
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=list(set(labels)),
            y=labels,
        )
        return torch.FloatTensor(class_weights)

    