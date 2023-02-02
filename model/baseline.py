import numpy as np
import torch
import timm
import copy
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
import torchvision.models as models
from torch.nn import functional as F

BACKBONES = {
    'resnet101',
    'resnet152',
    'dense121',
    'dense161',
    'dense201',
    'tf_efficientnet_b0',
    'tf_efficientnet_b1',
    'tf_efficientnet_b2',
    'vit_tiny_patch16_224',
    'vit_small_patch16_224',
    'vit_base_patch16_224',
    'vit_large_patch16_224',
}


class Baseline(nn.Module):
    def __init__(
            self,
            backbone,
            num_classes,
            pretrained=False,
            pretrained_path=None,
    ):
        super().__init__()
        # num_classes always keep 5, for the messido dataset, consider to change all 5 to 4, or just keep it.
        self.backbone = timm.create_model(backbone, pretrained=pretrained,num_classes=num_classes)
        if pretrained_path is not None:
            self.backbone.load_state_dict(torch.load(pretrained_path))

    def forward(self, x):
        x = self.backbone(x)
        return x
        