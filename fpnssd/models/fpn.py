import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


def resnet_by_layer(resnet_layer, pretrained=False):
    assert resnet_layer in [18, 34, 50, 101, 152]
    if resnet_layer == 18:
        net = resnet.resnet18(pretrained)
    elif resnet_layer == 34:
        net = resnet.resnet34(pretrained)
    elif resnet_layer == 50:
        net = resnet.resnet50(pretrained)
    elif resnet_layer == 101:
        net = resnet.resnet101(pretrained)
    elif resnet_layer == 152:
        net = resnet.resnet152(pretrained)
    else:
        raise ValueError
    return net


def _upsample_add(x, y):
    """Upsample and add two feature maps.
    Args:
      x: (Variable) top feature map to be upsampled.
      y: (Variable) lateral feature map.
    Returns:
      (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    """
    _, _, h, w = y.size()
    return F.upsample(x, size=(h, w), mode='bilinear', align_corners=False) + y


class ResNetFPN(nn.Module):
    def __init__(self, num_layers, pretrained=True):
        super().__init__()
        self.encoder = resnet_by_layer(num_layers, pretrained)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        # self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        # self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        # self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.top_conv = nn.Conv2d(2048, 256, kernel_size=1)

        self.lateral_conv1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(256, 256, kernel_size=1)

        self.smooth_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    # def _make_lateral_conv(self):
    #     layers = []
    #     for i in range(len(self.backbone_strides)):
    #         layers.append(nn.Conv2d(int(2048 / (i + 1)), 256, kernel_size=1))
    #     return layers
    #
    # def _make_smooth_conv(self):
    #     layers = []
    #     for i in range(len(self.backbone_strides)):
    #         layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
    #     return layers

    def forward(self, x):
        c1 = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=3, stride=2, padding=1)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        p6 = self.conv6(c5) # 1/64
        # p7 = self.conv7(F.relu(p6)) # 1/128
        # p8 = self.conv8(F.relu(p7)) # 1/256
        # p9 = self.conv9(F.relu(p8)) # 1/512

        p5 = self.top_conv(c5) # 1/32
        p4 = self.smooth_conv1(_upsample_add(p5, self.lateral_conv1(c4))) # 1/16
        p3 = self.smooth_conv2(_upsample_add(p4, self.lateral_conv2(c3))) # 1/8
        p2 = self.smooth_conv3(_upsample_add(p3, self.lateral_conv3(c2))) # 1/4

        feature_pyramid = [p2, p3, p4, p5, p6]
        return feature_pyramid
