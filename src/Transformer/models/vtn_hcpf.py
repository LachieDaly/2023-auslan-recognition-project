import torch
from torch import nn

from .common import FeatureExtractor, LinearClassifier, SelfAttention

class MMTensorNorm(nn.Module):
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
    
class VTNHCPF(nn.Module):
    def __init__(self, num_classes=29, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        """
        Initialises the Video Transformer Network with images of both hands and poseflow

        :param num_classes: number of classes to classify
        :param num_heads: number of heads in the multi-headed attention
        :param num_layers: number of layers in the multi-head attention
        :param embed_size: the feature size coming from the cnn feature extractror
        :param sequence_length: the video sequence length
        :param cnn: the name of the pretrained cnn to be used
        :param freeze_layers: number of layers starting from the input to freeze in the feature extracting cnn
        :param dropout: the dropout float to use throughout the model
        :param **kwargs: any other potential arguments
        :returns: Video Transformer Hand Crop Pose Flow model
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes

        self.feature_extractor = FeatureExtractor(cnn, embed_size, freeze_layers)

        num_attn_features = 2 * embed_size
        self.norm = MMTensorNorm(-1)
        self.bottle_mm = nn.Linear(106 + num_attn_features, num_attn_features)

        self.self_attention_decoder = SelfAttention(num_attn_features, num_attn_features,
                                                    [num_heads] * num_layers,
                                                    self.sequence_length, layer_norm=True)
    
        self.classifier = LinearClassifier(num_attn_features, num_classes, dropout)


    # When done like this it seems so simple :)
    def forward(self, mm_clip):
        """Extract the image feature vectors."""
        rgb_clip, pose_clip = mm_clip

        # Reshape to put both hand crops on the same axis.
        b, t, x, c, h, w = rgb_clip.size()
        rgb_clip = rgb_clip.view(b, t * x, c, h, w) # help
        z = self.feature_extractor(rgb_clip)
        # reshape back to extract features of both wrist crops as one feature vector
        z = z.view(b, t, -1)

        zp = torch.cat((z, pose_clip), dim=-1)

        zp = self.norm(zp)
        zp = self.bottle_mm(zp)
        zp = torch.nn.functional.relu(zp).clone()
        zp = self.self_attention_decoder(zp)

        y = self.classifier(zp)

        return y.mean(1)
    