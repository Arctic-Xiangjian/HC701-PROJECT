from torch.utils.data import ConcatDataset
import torch

from sklearn.utils.class_weight import compute_class_weight

import numpy as np


class WeightedConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(WeightedConcatDataset, self).__init__(datasets)
        self.weights = self.calculate_weights()
        
    def calculate_weights(self):
        labels = []
        for dataset in self.datasets:
            labels.extend([sample[1] for sample in dataset])
        labels = list(np.array(labels).flatten())
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=list(set(labels)),
            y=labels,
        )
        return torch.FloatTensor(class_weights)
        
    def __getitem__(self, idx):
        sample, label = super(WeightedConcatDataset, self).__getitem__(idx)
        return sample, label
