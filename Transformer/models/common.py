"""
Common model code

originally based on https://github.com/m-decoster/ChaLearn-2021-LAP/blob/master/src/models/common.py
and modified for this project
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.model import resnet18, resnet34

class FeatureExtractor(nn.Module):
    """Feature extractor for RGB clips, powered by a 2D CNN backbone."""

    def __init__(self, cnn="rn34", embed_size=512, freeze_layers=0):
        """Initialize the feature extractor with given CNN backbone and desired feature size"""
        super().__init__()

        if cnn == "rn18":
            model = resnet18(pretrained=True)
        elif cnn == "rn34":
            model = resnet34(pretrained=True)
        else:
            raise ValueError(f"Unkown value for `cnn`: {cnn}")
        
        # Get a number of layers from the resnet
        self.resnet = nn.Sequential(*list(model.children())[:-2])

        # Freeze layers if requests
        for layer_index in range(freeze_layers):
            for param in self.resnet[layer_index].parameters(True):
                param.required_grad_(False)

        if embed_size != 512:
            self.pointwise_conv = nn.Conv2d(512, embed_size, 1)
        else:
            self.pointwise_conv = nn.Identity()

    def forward(self, rgb_clip):
        """Extract features from the RGB images."""
        # b = batch ?
        # t = time ?
        # c = channel ?
        # h = height ?
        # w = width ?
        b, t, c, h, w = rgb_clip.size()
        # Process all sequential data in parallel as a large mini-batch.
        rgb_clip = rgb_clip.view(b * t, c, h, w)

        features = self.resnet(rgb_clip)

        # Transform to the desired embedding size.
        features = self.pointwise_conv(features)

        # Transform the output of the ResNet (C x H x W) to a single feature vector using pooling
        features= F.adaptive_avg_pool2d(features, 1).squeeze()

        # Restore the original dimensions of the tensor
        features = features.view(b, t, -1)

        return features
    
class SelfAttention(nn.Module):
    """Process sequences using self attention"""
    
    def __init__(self, input_size, hidden_size, n_heads, sequence_size, inner_hidden_factor=2, layer_norm=True):
        super().__init__()

        input_sizes = [hidden_size] * len(n_heads)
        input_sizes[0] = input_size
        hidden_sizes = [hidden_size] * len(n_heads)

        self.position_encoding = PositionEncoding(sequence_size, hidden_size)

        self.layers = nn.ModuleList([
            DecoderBlock()
        ])


# Private if we could


class Bottle(nn.Module):
    # TODO Learn waht is happening here
    """Perform the reshape routine before and after an operation."""
    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)
    
class BottleSoftmax(Bottle, nn.Softmax):
    """Perform the reshape routine before and after a softmax operation."""
    pass

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self, d_model, attn_dropout):
