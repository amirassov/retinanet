import os
import torch
import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .resnext import ResNeXt
from .se_module import SEBlock

__all__ = ['se_resnext50', 'se_resnext101', 'se_resnext101_64', 'se_resnext152']

model_urls = {
    'se_resnext50': 'https://nizhib.ai/share/pretrained/se_resnext50-5cc09937.pth'
}


class SEBottleneck(nn.Module):
    """
    SE-RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None,
                 reduction=16):
        super(SEBottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride,
                               padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * 4, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnext50(num_classes=1000, pretrained=False):
    """Constructs a SE-ResNeXt-50 model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnext50']))
    return model


def se_resnext101(num_classes=1000, pretrained=True):
    """Constructs a SE-ResNeXt-101 (32x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 4, 23, 3], num_classes=num_classes)

    if pretrained:
        path = os.path.join(os.getenv("HOME"), '.torch/models/se_resnext101_best.pth')
        pretrained_parallel_state_dict = torch.load(path, map_location='cpu')['state_dict']
        pretrained_normal_state_dict = dict()
        for key in pretrained_parallel_state_dict.keys():
            # key = key
            pretrained_normal_state_dict[key.split('module.')[1]] = \
            pretrained_parallel_state_dict[key]
        model.load_state_dict(pretrained_normal_state_dict)

    return model


def se_resnext101_64(num_classes=1000):
    """Constructs a SE-ResNeXt-101 (64x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 64, [3, 4, 23, 3], num_classes=num_classes)
    return model


def se_resnext152(num_classes=1000):
    """Constructs a SE-ResNeXt-152 (32x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 8, 36, 3], num_classes=num_classes)
    return model
