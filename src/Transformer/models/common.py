"""
Common model code

originally based on https://github.com/m-decoster/ChaLearn-2021-LAP/blob/master/src/models/common.py
and modified for this project
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import LSTM
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights

class FeatureExtractor(nn.Module):
    """Feature extractor for RGB clips, powered by a 2D CNN backbone."""

    def __init__(self, cnn="rn34", embed_size=512, freeze_layers=0):
        """
        __init__ Initialize the feature extractor with given CNN backbone and desired feature size

        :param cnn: which cnn will be used to extract features
        :param embed_size: the expected output feature size
        :param freeze_layers: number of layers to freeze weights
        :return: calibrated feature extractor
        """
        super().__init__()

        if cnn == "rn18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif cnn == "rn34":
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
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
        features = F.adaptive_avg_pool2d(features, 1).squeeze()

        # Restore the original dimensions of the tensor
        features = features.view(b, t, -1)

        return features
    
class SelfAttention(nn.Module):
    """Process sequences using self attention"""
    
    def __init__(self, input_size, hidden_size, n_heads, sequence_size, inner_hidden_factor=2, layer_norm=True):
        """
        __init__ intialised our SelfAttention layer

        :param input_size: input size into layer
        :param hidden_size: hidden size inside decoder block
        :param n_heads: number of heads in each layer
        :param sequence_size: the number of feature frames to be parsed to the mdoel
        :param inner_hidden_factor: number of hidden layers
        :param layer_norm: bool to determine whether to use layer normalisation
        :return: calibrated SelfAttention Layer
        """
        super().__init__()

        # input size (2 * embed size even with poseflow features because of bottle)
        # hidden size is also 2 * embed size
        input_sizes = [hidden_size] * len(n_heads)
        input_sizes[0] = input_size # Of course the first layer needs to accept the right size
        hidden_sizes = [hidden_size] * len(n_heads)

        self.position_encoding = PositionEncoding(sequence_size, hidden_size)

        self.layers = nn.ModuleList([
            # Because we have two layers
            # we get two DecoderBlocks with 
            # input_size = in_size = 512
            # hidden_size = hid_size = 512
            # inner_hidden_size = hid_size * inner_hidden_factor = 1024
            # n_head = n_head = 4
            # d_k = hid_size // n_head = 128
            # d_v = hid_size // n_head = 128
            DecoderBlock(inp_size, hid_size, hid_size * inner_hidden_factor, n_head, hid_size // n_head,
                          hid_size // n_head, layer_norm=layer_norm)
            for i, (inp_size, hid_size, n_head) in enumerate(zip(input_sizes, hidden_sizes, n_heads))
        ])

    def forward(self, x):
        outputs, attentions = [], []
        x = self.position_encoding(x)

        # Are we even using outputs here?
        # would removing it change anything
        for layer in self.layers:
            x, attn = layer(x)
            outputs.append(x)
        return x
    

# LSTM Stuffs
class LongShortTermMemory(nn.Module):
    """Process sequences using lstm"""

    def __init__(self, input_size, hidden_size, layer_norm=True):
        """
        __init__ initialises the LongShortTermMemory network

        :param input_size: input size into the network
        :param hidden_size: the hidden layer size in the netowrk
        :param layer_norm: whether to use layer normalisation
        :return: LongShortTermMemory network 
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        # sequence size needed?
        self.lstm = LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        return self.lstm(x)


class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0):
        """
        Initialises linear classifier

        :param input_size: input dimension into linear classifier
        :param num_classes: number of classes to classify
        :param 
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, num_classes)

        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        return self.fc(self.dropout(x))

# Private if we could

class Bottle(nn.Module):
    # TODO Learn what is happening here
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
    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax()

    def forward(self, q, k, v):
        # batch matrix multiplication
        # Calculates what we should be paying attention to
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # now we figure out what we should be paying attention to?
        output = torch.bmm(attn, v)
        return output, attn
    
class LayerNormalisation(nn.Module):
    """Layer normalisation module"""
    def __init__(self, d_hid, eps=1e-3):
        """
        __init__ initialises the normalisation layer

        :param d_hid:
        :param eps: 
        :return: normalisation layer
        """
        super(LayerNormalisation, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        # layer normalisation algorithm
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""
    def __init__(self, n_head, input_size, output_size, d_k, d_v, dropout=0.1, layer_norm=True):
        """
        :param n_head: Number of attention heads
        :param input_size: Input feature size
        :param output_size: Output feature size
        :param d_k: key feature size for each head?
        :param d_v: value feature size for each head?
        dropout: Dropout rate after projection
        """
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        # Create parameter of shape (n_head, input_size, features_size)
        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, input_size, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, input_size, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, input_size, d_v))

        self.attention = ScaledDotProductAttention(input_size)
        self.layer_norm = LayerNormalisation(input_size) if layer_norm else nn.Identity()

        self.dropout = nn.Dropout(dropout)

        # Fills the input tensor
        nn.init.xavier_normal_(self.w_qs)
        nn.init.xavier_normal_(self.w_ks)
        nn.init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # query, key, value
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)

        # treat the result as a (n_head * mb_size) size batch
        # matrix multiply q,k,v with weights
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)

        # pass all values
        outputs, attns = self.attention(q_s, k_s, v_s)

        # interpret this
        split_size = mb_size.item() if isinstance(mb_size, torch.Tensor) else mb_size
        h, t, e = outputs.size()
        outputs = outputs.view( h // split_size, split_size, t, e) # (B x T x H*E)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(split_size, len_q, -1)

        outputs = self.dropout(outputs)
        
        return self.layer_norm(outputs + residual), attns

class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_hid, d_inner_hid, dropout=0.1, layer_norm=True):
        """
        Initialises Positionwise Feed Forward layer

        :param d_hid: dimensions of input and output of first and second conv1d layers resepectively
        :param d_inner_hid: dimension of output and input of first and second conv1d layers respectively
        :param dropout: dropout value
        :param layer_norm: whether to perform layer normalisation
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1) # position-wise
        self.layer_norm = LayerNormalisation(d_hid) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1,2)))
        output = self.w_2(output).transpose(2,1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)

class DecoderBlock(nn.Module):
    """Compose with two layer"""
    def __init__(self, input_size, hidden_size, inner_hidden_size, n_head, d_k, d_v, dropout=0.1, layer_norm=True):
        """
        Initialises the DecoderBlocks used in the model

        :param input_size: input size of the decoder block
        :param hidden_size: hidden layer size of the MultiHeadAttention and input to PositionwiseFeedForward
        :param inner_hidden_size: hidden size of PositionFeedForward network
        :param n_head: number of heads in the MultiHeadAttention layer
        :param d_k: feature size for one hand
        :param d_v: feature size for one hand
        :param dropout: dropout for Multi Head Attention and Position Feed Forward layers
        :param layer_norm: whether to perform layer normalisation 
        """
        super(DecoderBlock, self).__init__()
        # Multi head attention layer
        self.slf_attn = MultiHeadAttention(n_head, input_size, hidden_size, d_k, d_v, dropout=dropout,
            layer_norm=layer_norm)
        
        self.pos_ffn = PositionwiseFeedForward(hidden_size, inner_hidden_size, dropout=dropout, 
                                               layer_norm=layer_norm)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class PositionEncoding(nn.Module):
    """
    Generate the positional encodings for the self attention network
    """
    def __init__(self, n_positions, hidden_size):
        """
        Initialises the PositionEncoding layer

        :param n_positions: the number of featue frames (video frames)
        :param hidden_size: the number of features in each feature frame
        :returns: Position Encoding Layer
        """
        super().__init__()
        # Table that stores embeddings of a fixed dictionary and size
        self.enc = nn.Embedding(n_positions, hidden_size, padding_idx=0)

        # n_positions = sequence length
        # hidden_size = 512 or 1024?
        # for n_position 3 and hidden
        # pairs are same values
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / hidden_size) for j in range(hidden_size)]
            if pos != 0 else np.zeros(hidden_size) for pos in range(n_positions)
        ])

        # The same values are passed either to sin or cos
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] - np.cos(position_enc[1:, 1::2]) # dim wi + 1

        self.enc.weight = torch.nn.Parameter(torch.from_numpy(position_enc).to(self.enc.weight.device, torch.float))

    def forward(self, x):
        indicies = torch.arange(0, x.size(1)).to(self.enc.weight.device, torch.long)
        encodings = self.enc(indicies)
        # The positional encodings are added to the feature vector
        x += encodings
        return x
