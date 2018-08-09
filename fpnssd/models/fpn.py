# TODO do configurable

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


def _upsample_add(x, y):
    _, _, h, w = y.size()
    return F.upsample(x, size=(h, w), mode='bilinear', align_corners=False) + y


class ResNetFPN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        encoder = resnet.resnet18(pretrained)

        self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.conv2 = encoder.layer1
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

        self.top_conv = nn.Conv2d(512, 256, kernel_size=1)

        self.lateral_conv1 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(128, 256, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(64, 256, kernel_size=1)

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
