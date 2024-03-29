import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torchvision.models.video import R2Plus1D_18_Weights
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
import torchvision
from functools import partial
from collections import OrderedDict
import math

import os,inspect,sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)

def convert_relu_to_swish(model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, nn.SiLU(True))
                # setattr(model, child_name, Swish())
            else:
                convert_relu_to_swish(child)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mul_(torch.sigmoid(x))

class r2plus1d_18(nn.Module):
    """
    Our R(2+1)D model that we are using
    Pretrained with the KINETICS400 dataset
    
    """
    def __init__(self, pretrained=True, num_classes=500, dropout_p=0.5):
        """
        initialises R(2+1)D model for RGB frames

        :param pretrained: if true, use pretrained KINETICS400 weights
        :param num_classes: number of classes to classify
        :param dropout: dropout probability
        """
        super(r2plus1d_18, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        if pretrained:
            model = torchvision.models.video.r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        else:
            model = torchvision.models.video.r2plus1d_18()
        modules = list(model.children())[:-1]
        self.r2plus1d_18 = nn.Sequential(*modules)
        convert_relu_to_swish(self.r2plus1d_18)
        self.fc1 = nn.Linear(model.fc.in_features, self.num_classes)
        self.dropout = nn.Dropout(dropout_p, inplace=True)


    def forward(self, x):
        out = self.r2plus1d_18(x)
        # Flatten the layer to fc
        out = out.flatten(1)
        out = self.dropout(out)
        out = self.fc1(out)
        return out

class flow_r2plus1d_18(nn.Module):
    def __init__(self, pretrained=False, num_classes=500, dropout_p=0.5):
        """
        Initialises R(2+1)D model for optical flow

        :param pretrained: if true, use pretrained KINETICS400 weights
        :param num_classes: number of classes to classify
        :param dropout: dropout probability
        """
        super(flow_r2plus1d_18, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        model = torchvision.models.video.r2plus1d_18(pretrained=self.pretrained)

        model.stem[0] = nn.Conv3d(2, 45, kernel_size=(1, 7, 7),
                            stride=(1, 2, 2), padding=(0, 3, 3),
                            bias=False)

        # delete the last fc layer
        modules = list(model.children())[:-1]
        self.r2plus1d_18 = nn.Sequential(*modules)
        convert_relu_to_swish(self.r2plus1d_18)
        self.fc1 = nn.Linear(model.fc.in_features, self.num_classes)
        self.dropout = nn.Dropout(dropout_p, inplace=True)

    def forward(self, x):
        out = self.r2plus1d_18(x)
        out = out.flatten(1)
        out = self.dropout(out)
        out = self.fc1(out)
        return out