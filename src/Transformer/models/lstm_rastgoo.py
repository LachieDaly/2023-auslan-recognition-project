import torch
from torch import nn

from .common import FeatureExtractor, LinearClassifier, LongShortTermMemory

class MMTensor(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, x):
        mean = torch.mean(x, dim=self.dim).unsqueeze(self.dim)
        std = torch.std(x, dim=self.dim).unsqueeze(self.dim)
        return (x - mean) / std
    
class LSTMR(nn.Module):
    def __init__(self, num_classes=29, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()

        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes
        
        self.feature_extractor = FeatureExtractor(cnn, embed_size, freeze_layers)
        num_features = embed_size * 2

        self.norm = MMTensor(-1)

        self.bottle_mm = nn.Linear(295 + num_features, num_features)

        self.lstm = LongShortTermMemory(num_features, num_features, True)

        self.classifier = LinearClassifier(num_features, num_classes, dropout)

    def forward(self, mm_clip):
        """Extract the image feature vectors."""
        rgb_clip, pose_clip = mm_clip

        # Reshape to put both hand crops on the same axis
        b, t, x, c, h, w = rgb_clip.size()
        rgb_clip = rgb_clip.view(b, t * x, c, h, w)
        z = self.feature_extractor(rgb_clip)
        # reshape back to extract features of both wrist crops as one feature vector
        z = z.view(b, t, -1)

        zp = torch.cat((z, pose_clip), dim=-1)

        zp = self.norm(zp)
        zp = self.bottle_mm(zp)
        zp = torch.nn.functional.relu(zp).clone()
        # h_n, c_n are the hidden features of the lstm
        zp, (h_n, c_n) = self.lstm(zp)

        # Select last output
        last_output = zp[:, -1, :]

        y = self.classifier(last_output)

        return y