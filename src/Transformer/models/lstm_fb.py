import torch
from torch import nn

from .common import FeatureExtractor, LinearClassifier, LongShortTermMemory

class MMTensor(nn.Module):
    def __init__(self, dim):
        """
        Initialises normalisation layer

        :param dim: dimesnion to unsqueeze and calculate mean and standard deviation across
        :returns: normalisation layer
        """
        super().__init__()

        self.dim = dim

    def forward(self, x):
        mean = torch.mean(x, dim=self.dim).unsqueeze(self.dim)
        std = torch.std(x, dim=self.dim).unsqueeze(self.dim)
        return (x - mean) / std
    
class LSTMFB(nn.Module):
    def __init__(self, num_classes=29, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0.1, **kwargs):
        """
        LSTMR intialises Long Short Term Memory Full Body model

        :param num_classes: number of classes to classify
        :param embed_size: the feature size coming from the cnn feature extractror
        :param sequence_length: the video sequence length
        :param cnn: the name of the pretrained cnn to be used
        :param freeze_layers: number of layers starting from the input to freeze in the feature extracting cnn
        :param dropout: the dropout float to use throughout the model
        :param **kwargs: any other potential arguments
        :returns: Long Short Term Memory Full body network
        """
        super().__init__()
        print(type(dropout))
        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes
        
        self.feature_extractor = FeatureExtractor(cnn, embed_size, freeze_layers)
        num_features = embed_size

        self.norm = MMTensor(-1)
        self.lstm = LongShortTermMemory(num_features, num_features, True)
        self.dropout = nn.Dropout(dropout)

        self.classifier = LinearClassifier(num_features, num_classes, dropout)

    def forward(self, rgb_clip):
        z = self.feature_extractor(rgb_clip)

        zp = self.norm(z)
        zp = torch.nn.functional.relu(zp).clone()
        zp, (h_n, c_n) = self.lstm(zp)
        zp = self.dropout(zp)
        # Select last output
        last_output = zp[:, -1, :]

        y = self.classifier(last_output)

        return y