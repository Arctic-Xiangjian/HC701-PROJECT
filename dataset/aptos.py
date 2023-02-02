import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from torchvision import utils
from PIL import Image
class APTOS(Dataset):
    def __init__(self, root_dir, transform=None, train=True, resize_shape=(1024, 1024)):
        self.root_dir = root_dir
        self.train = train
        self.train_image_path = os.path.join(self.root_dir, 'hc701-fed_data/aptos/train_images')
        self.test_image_path = os.path.join(self.root_dir, 'hc701-fed_data/aptos/test_images')
        self.train_csv_path = os.path.join(self.root_dir, 'hc701-fed_data/aptos/train.csv')
        self.test_csv_path = os.path.join(self.root_dir, 'hc701-fed_data/aptos/test.csv')
        self.resize_shape = resize_shape

        # Use transforms.Resize and transforms.ToTensor to resize and convert image to tensor
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_shape),
            transforms.ToTensor(),
        ]) if transform is None else transform

        self.data = None
        if self.train:
            train_data = pd.read_csv(self.train_csv_path)
            train_data['id_code'] = train_data['id_code'].apply(lambda x: x + '.png')
            train_data['diagnosis'] = train_data['diagnosis'].astype(int)
            self.data = train_data
        else:
            test_data = pd.read_csv(self.test_csv_path)
            test_data['id_code'] = test_data['id_code'].apply(lambda x: x + '.png')
            self.data = test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.train_image_path if self.train else self.test_image_path, self.data.iloc[idx, 0])
        image = self._read_image(img_name)
        if self.train:
            label = self.data.iloc[idx, 1]
            return image, label
        return image

    def _read_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image
