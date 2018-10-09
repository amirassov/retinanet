# TODO do configurable

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


def _upsample_add(x, y):
    _, _, h, w = y.size()
    return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False) + y

def get_channels(architecture):
    if architecture in ['resnet18', 'resnet34']:
        return [512, 256, 128, 64]
    elif architecture in ['resnet50', 'resnet101', 'resnet152']:
        return [2048, 1024, 512, 256]
    else:
        raise Exception(f'{architecture} is not supported as backbone')


class ResNetFPN(nn.Module):
    def __init__(self, pretrained=True, architecture='resnet18'):
        super().__init__()
        encoder = getattr(resnet, architecture)(pretrained)
        channles = get_channels(architecture)

        self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.conv2 = encoder.layer1
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

        self.top_conv = nn.Conv2d(channles[0], 256, kernel_size=1)

        self.lateral_conv1 = nn.Conv2d(channles[1], 256, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(channles[2], 256, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(channles[3], 256, kernel_size=1)

        self.smooth_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        p5 = self.top_conv(c5)  # 1/32
        p4 = self.smooth_conv1(_upsample_add(p5, self.lateral_conv1(c4)))  # 1/16
        p3 = self.smooth_conv2(_upsample_add(p4, self.lateral_conv2(c3)))  # 1/8
        p2 = self.smooth_conv3(_upsample_add(p3, self.lateral_conv3(c2)))  # 1/4

        feature_pyramid = [p2, p3, p4, p5]
        return feature_pyramid
