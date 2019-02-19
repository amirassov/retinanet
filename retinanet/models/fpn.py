import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
import pydoc


def _upsample_add(x, y):
    _, _, h, w = y.size()
    return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False) + y


class ResNetBackbone:
    def __init__(self, pretrained=True, architecture='resnet18'):
        self.pretrained = pretrained
        encoder = getattr(resnet, architecture)(pretrained=pretrained)
        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.channels = [
            self.layer4[-1].conv2.out_channels,
            self.layer3[-1].conv2.out_channels,
            self.layer2[-1].conv2.out_channels,
            self.layer1[-1].conv2.out_channels]


class RetinaNetFPN(nn.Module):
    def __init__(self, backbone_path=None, backbone_params=None):
        super().__init__()
        if backbone_path is None:
            backbone = ResNetBackbone(**backbone_params)
        else:
            backbone = pydoc.locate(backbone_path)(**backbone_params)

        self.conv1 = backbone.layer0
        self.conv2 = backbone.layer1
        self.conv3 = backbone.layer2
        self.conv4 = backbone.layer3
        self.conv5 = backbone.layer4
        self.top_conv = nn.Conv2d(backbone.channels[0], 256, kernel_size=1)

        self.lateral_conv1 = nn.Conv2d(backbone.channels[1], 256, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(backbone.channels[2], 256, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(backbone.channels[3], 256, kernel_size=1)

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
