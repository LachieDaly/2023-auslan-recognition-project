import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
# groups == joints_number
def channel_shuffle(x, groups):
    """
    
    """
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # So groups and channels per group are transposed
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class swish(nn.Module):
    def __init__(self):
        """
        initialises swish (sigmoid) layer
        """
        super(swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)
    
class Attention(nn.Module):
    def __init__(self, channel, mid, groups):
        """
        Initialises attention layer

        :param channel: channels into first convolutional layer
        :param mid: exit from first conv layer and input to second
        :param groups: number of groups
        """
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, mid, 1, padding=0, groups=groups, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, channel, 1, padding=0, groups=groups, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

        
def conv1x1(in_channels, out_channels, groups=133):
    """
    Returns 1x1 convolutional layer with padding

    :param in_channels: number of incoming channels
    :param out_channels: number of outgoing channels
    :param groups: Normal pointwise convolution When groups == 1, 
                   Grouped pointwise convolution when groups > 1
    :return: configured 1x1 Conv2D layer
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1,
        padding=0)
        
def conv3x3(in_channels, out_channels, groups=1,stride=1):
    """
    Returns 3x3 convolution with padding

    :param in_channels: number of incoming channels
    :param out_channels: number of outgoing channels
    :param groups: Normal pointwise convolution When groups == 1
                   Grouped pointwise convolution when groups > 1
    :return: configured 3x3 Conv2D layer
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        groups=groups,
        stride=stride,
        padding=1)

def get_inplanes():
    """
    return list of inplane
    """
    return [64, 128, 256, 512]

class TemporalDownsampleBlock(nn.Module):
    def __init__(self, in_planes, planes, joints_number, mid, relu=True):
        """
        Initialise Temporal Downsample Block

        :param in_planes: in channels to 1x1 convolution
        :param planes: out channels from the 1x1 convolution
        :param joints_number: number of skeleton joints
        :param mid: 
        """
        super().__init__()
        # in_planes and planes are both n*joints_number, n is integer
        self.joints_number = joints_number
        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            in_planes,
            planes,
            self.joints_number,
            mid,
            batch_norm=True,
            relu=relu
        )

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups, mid, batch_norm=True, relu=False):
        """
        Initialises grouped conv1x1 layer

        :param in_channels: channels into convolutional layer
        :param out_channels: channels out of convolutional layer
        :param groups: groups into conv layer
        :param mid: middle channels of attention
        :param batch_norm: if True, peform batch normalisation
        :param relu: if True, add swish layer
        """
        modules = OrderedDict()
        attention = Attention(in_channels, mid, groups=groups)
        modules['attention'] = attention
        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['swish'] = swish()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):
        out = self.g_conv_1x1_compress(x)
        return out
        
class FrameDownsampleBlock(nn.Module):

    def __init__(self, in_planes, planes, frames, mid, relu=True, stride=1):
        """
        Initialise Frame Downsample Block

        :param in_planes: input channels into 3x3 conv layer
        :param planes: output channels from 3x3 conv layer
        :param frames: number of frames (groups)
        :param mid: mid channels in attention
        :param relu: if True, add swish layer
        :param stride: stride of convolutional layers
        :return: Frame Downsample Block
        """
        super().__init__()
        # in_planes and planes are both n*joints_number, n is integer
        self.frames = frames

        self.g_conv_3x3_compress = self._make_grouped_conv3x3(
            in_planes,
            planes,
            self.frames,
            mid,
            stride,
            batch_norm=True,
            relu=relu
        )

    def _make_grouped_conv3x3(self, in_channels, out_channels, groups, mid, stride, batch_norm=True, relu=False):
        """
        Build 3x3 conv block

        :param in_channels: input channels into 3x3 conv layer
        :param out_channels: output channels from 3x3 conv layer
        :param groups: number of groups
        :param mid: mid channels in attention
        :param stride: stride of convolutional layers
        :param batch_norm: if True, perform batch normalisation
        :param relu: if True, add swish layer
        :return: Frame Downsample Block
        """
        modules = OrderedDict()
        attention = Attention(in_channels, mid, groups=groups)
        modules['attention'] = attention
        conv = conv3x3(in_channels, out_channels, groups=groups,stride=stride)
        modules['conv3x3'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['swish'] = swish()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv
            
    def forward(self, x):
        out = self.g_conv_3x3_compress(x)
        return out

class T_Pose_model(nn.Module):

    def __init__(self, frames_number, joints_number, n_classes=29):
        """
        Initialise SSTCN model

        :param frames_number: number of video frames
        :param joint_number: number of skeleton joints
        :param n_classes: number of classes to classify
        :return: SSTCN model
        """
        super().__init__()
        self.in_channels = frames_number
        self.joints_number = joints_number
        self.final_frames_number = frames_number

        self.bn = nn.BatchNorm2d(self.in_channels * self.joints_number)

        self.swish = swish()

        # Temporal Downsamples
        self.t1downsample = TemporalDownsampleBlock(self.in_channels, 
                                                    frames_number, 
                                                    1, 
                                                    10)
        
        self.t2downsample = TemporalDownsampleBlock(frames_number, 
                                                    self.final_frames_number, 
                                                    1, 
                                                    10, 
                                                    relu=False)

        # Spatial Downsamples
        self.f1downsample = FrameDownsampleBlock(self.final_frames_number * joints_number,
                                                 self.final_frames_number * joints_number,
                                                 joints_number, 
                                                 joints_number * 10)

        self.f2downsample = FrameDownsampleBlock(self.final_frames_number * joints_number,
                                                 self.final_frames_number*joints_number,
                                                 joints_number,joints_number * 10,
                                                 relu=False)

        self.f3downsample = FrameDownsampleBlock(self.final_frames_number * joints_number,
                                                 self.final_frames_number * joints_number, 
                                                 self.final_frames_number, 
                                                 self.final_frames_number * 10)

        self.f4downsample = FrameDownsampleBlock(self.final_frames_number * joints_number,
                                                 self.final_frames_number * joints_number, 
                                                 self.final_frames_number, 
                                                 self.final_frames_number * 10, 
                                                 relu=False)

        self.f5downsample = FrameDownsampleBlock(self.final_frames_number * joints_number,
                                                 self.final_frames_number * joints_number, 
                                                 1, 
                                                 10 * 10)

        self.f6downsample = FrameDownsampleBlock(self.final_frames_number * joints_number, 
                                                 30 * joints_number, 
                                                 1,
                                                 10 * 10)
        
        self.dropout = nn.Dropout2d(0.333)

        self.fc1 = nn.Linear(990, n_classes)

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        x = self.bn(x)
        x = self.swish(x)
        res = x
        x = x.view(-1, self.in_channels, self.joints_number * height, width)
        x = self.t1downsample(x)
        x = self.t2downsample(x)
        x = x.view(-1, self.final_frames_number * self.joints_number, height, width)
        x = res + x
        x = self.swish(x)
        res = x

        x = channel_shuffle(x, self.final_frames_number)
        x = self.f1downsample(x)
        x = self.dropout(x)
        x = self.f2downsample(x)

        x = channel_shuffle(x, self.joints_number)
        x = x + res
        x = self.swish(x)
        res = x
        x = self.f3downsample(x)
        x = self.dropout(x)
        x = self.f4downsample(x)
        x = x + res
        x = self.swish(x)

        x = self.f5downsample(x)
        x = self.dropout(x)
        x = self.f6downsample(x)
        x = F.avg_pool2d(x, x.data.size()[-2:])
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def init_weights(self):
        """
        Initialise the SSTCN model weights
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class LabelSmoothingCrossEntropy(nn.Module):
    """
    This label smoothing should be imported because its used so much
    """
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()