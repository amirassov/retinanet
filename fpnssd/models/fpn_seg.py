import os
import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from zoo import inplaceabn, dpn
from zoo.inplaceabn.modules import InPlaceABNSync, InPlaceABN

torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
wider_resnet_path = {'wider_resnet38': os.path.join(model_dir, 'wide_resnet38_ipabn_lr_256.pth.tar')}
resnext_path = {'resnext50': os.path.join(model_dir, 'resnext50_inplace.pth.tar'),
                'resnext101': os.path.join(model_dir, 'resnext101_ipabn_lr_512.pth.tar'),
                'resnext152': os.path.join(model_dir, 'resnext152_ipabn_lr_256.pth.tar')}


class FPNConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        return x

class FPNConvBNBlock(nn.Module):
    def __init__(self, input_channels, output_channels, norm_act):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_act(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_act(output_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class FPNModule(nn.Module):
    def __init__(self,
                 layer_sizes,
                 fpn_features=512,
                 fpn_conv_features=256,
                 upsampling='nearest'
                 ):
        super().__init__()
        self.upsampling = upsampling
        self.lateral5 = nn.Conv2d(layer_sizes[2], fpn_features, kernel_size=1, stride=1, padding=0)

        self.lateral4 = nn.Conv2d(layer_sizes[1], fpn_features, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(fpn_features, fpn_features, kernel_size=3, stride=1, padding=1)

        self.lateral3 = nn.Conv2d(layer_sizes[0], fpn_features, kernel_size=1, stride=1, padding=0)
        self.p3 = nn.Conv2d(fpn_features, fpn_features, kernel_size=3, stride=1, padding=1)

        self.p6 = nn.Conv2d(layer_sizes[2], fpn_features, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Conv2d(fpn_features, fpn_features, kernel_size=3, stride=2, padding=1)
        for i in range(3, 8):
            self.add_module("level{}".format(i),
                            FPNConvBlock(input_channels=fpn_features, output_channels=fpn_conv_features))

    def forward(self, x):
        conv3, conv4, conv5 = x
        p5 = self.lateral5(conv5)
        p5_upsampled = F.upsample(p5, size=None, scale_factor=2)

        p4 = self.lateral4(conv4)
        p4 = p5_upsampled + p4
        p4_upsampled = F.upsample(p4, size=None, scale_factor=2)

        p3 = self.lateral3(conv3)
        p3 = p4_upsampled + p3

        p3 = nn.ReLU()(self.p3(p3))
        p4 = nn.ReLU()(self.p4(p4))
        p5 = nn.ReLU()(p5)
        p6 = nn.ReLU()(self.p6(conv5))
        p7 = nn.ReLU()(self.p7(p6))
        # multi-scale context is aggregated through upsampling and concatenation
        outputs = [
            self.level3(p3),
            F.upsample(self.level4(p4), scale_factor=2, mode=self.upsampling),
            F.upsample(self.level5(p5), scale_factor=4, mode=self.upsampling),
            F.upsample(self.level6(p6), scale_factor=8, mode=self.upsampling),
            F.upsample(self.level7(p7), scale_factor=16, mode=self.upsampling),
        ]

        return torch.cat(outputs, dim=1)

class FPNModuleBN(nn.Module):
    def __init__(self,
                 layer_sizes,
                 norm_act,
                 fpn_features=384,
                 fpn_conv_features=192,
                 upsampling='nearest'
                 ):
        super().__init__()
        self.upsampling = upsampling
        self.lateral7 = nn.Conv2d(fpn_features, fpn_features, kernel_size=1, stride=1, padding=0)
        self.lateral6 = nn.Conv2d(fpn_features, fpn_features, kernel_size=1, stride=1, padding=0)
        self.lateral5 = nn.Conv2d(layer_sizes[2], fpn_features, kernel_size=1, stride=1, padding=0)
        self.lateral4 = nn.Conv2d(layer_sizes[1], fpn_features, kernel_size=1, stride=1, padding=0)
        self.lateral3 = nn.Conv2d(layer_sizes[0], fpn_features, kernel_size=1, stride=1, padding=0)
        self.p6 = nn.Conv2d(layer_sizes[2], fpn_features, kernel_size=3, stride=2, padding=1)
        self.conv6_act = norm_act(fpn_features)
        self.p7 = nn.Conv2d(fpn_features, fpn_features, kernel_size=3, stride=2, padding=1)
        self.conv7_act = norm_act(fpn_features)
        for i in range(3, 8):
            self.add_module("level{}".format(i),
                            FPNConvBNBlock(input_channels=fpn_features, output_channels=fpn_conv_features, norm_act=norm_act))

    def forward(self, x):
        conv3, conv4, conv5 = x

        conv6 = self.p6(conv5)
        conv6 = self.conv6_act(conv6)

        conv7 = self.p7(conv6)
        conv7 = self.conv7_act(conv7)

        p7 = self.lateral7(conv7)
        p7_upsampled = F.upsample(p7, size=None, scale_factor=2)

        p6 = self.lateral6(conv6)
        p6 = p7_upsampled + p6
        p6_upsampled = F.upsample(p6, size=None, scale_factor=2)

        p5 = self.lateral5(conv5)
        p5 = p6_upsampled + p5
        p5_upsampled = F.upsample(p5, size=None, scale_factor=2)

        p4 = self.lateral4(conv4)
        p4 = p5_upsampled + p4
        p4_upsampled = F.upsample(p4, size=None, scale_factor=2)

        p3 = self.lateral3(conv3)
        p3 = p4_upsampled + p3
        # multi-scale context is aggregated through upsampling and concatenation
        outputs = [
            self.level3(p3),
            F.upsample(self.level4(p4), scale_factor=2, mode=self.upsampling),
            F.upsample(self.level5(p5), scale_factor=4, mode=self.upsampling),
            F.upsample(self.level6(p6), scale_factor=8, mode=self.upsampling),
            F.upsample(self.level7(p7), scale_factor=16, mode=self.upsampling),
        ]

        return torch.cat(outputs, dim=1)


class FPNResneXt(nn.Module):
    def __init__(self,
                 seg_classes=21,
                 fpn_features=512,
                 fpn_conv_features=256,
                 downsampling=1,
                 fc_channels=1024,
                 backbone_arch='resnext101',
                 upsample_size=None,
                 **model_params):
        super().__init__()
        self.mask_downsampling = downsampling
        self.upsample_size = upsample_size

        layer_sizes = [512, 1024, 2048]
        self.fpn_module = FPNModule(layer_sizes=layer_sizes, fpn_features=fpn_features,
                                    fpn_conv_features=fpn_conv_features)

        self.classifier = nn.Sequential(
            nn.Conv2d(5 * fpn_conv_features, fc_channels, kernel_size=1, padding=1),
            model_params['norm_act'](fc_channels),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(fc_channels, seg_classes, kernel_size=1),
        )
        init_weights(self)
        self.encoder = inplaceabn.models.__dict__["net_" + backbone_arch](**model_params)
        state_dict = torch.load(resnext_path[backbone_arch])['state_dict']
        self.encoder.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()},
                                     strict=True)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        out = self.encoder.mod1(x)
        out = self.encoder.mod2(out)
        out = self.encoder.mod3(out)
        c3 = out
        out = self.encoder.mod4(out)
        c4 = out
        out = self.encoder.mod5(out)
        out = self.encoder.bn_out(out)
        c5 = out

        x = self.fpn_module([c3, c4, c5])
        if self.upsample_size:
            x = F.upsample(self.classifier(x), mode='bilinear',
                           size=(self.upsample_size[0], self.upsample_size[1]))
        else:
            x = F.upsample(self.classifier(x), mode='bilinear',
                           size=(h // self.mask_downsampling, w // self.mask_downsampling))
        return x

class FPNResneXtMapillary(nn.Module):
    def __init__(self,
                 seg_classes=65,
                 fpn_features=512,
                 fpn_conv_features=256,
                 downsampling=1,
                 fc_channels=1280,
                 backbone_arch='resnext101',
                 upsample_size=None,
                 **model_params):
        super().__init__()
        self.mask_downsampling = downsampling
        self.upsample_size = upsample_size

        layer_sizes = [512, 1024, 2048]
        self.fpn_module = FPNModuleBN(layer_sizes=layer_sizes, fpn_features=fpn_features,
                                      fpn_conv_features=fpn_conv_features, norm_act=model_params['norm_act'])

        self.classifier = nn.Sequential(
            nn.Conv2d(5 * fpn_conv_features, fc_channels, kernel_size=1, padding=1),
            model_params['norm_act'](fc_channels),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(fc_channels, seg_classes, kernel_size=1),
        )
        init_weights(self)
        self.encoder = inplaceabn.models.__dict__["net_" + backbone_arch](**model_params)
        state_dict = torch.load(resnext_path[backbone_arch])['state_dict']
        self.encoder.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()},
                                     strict=True)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        out = self.encoder.mod1(x)
        out = self.encoder.mod2(out)
        out = self.encoder.mod3(out)
        c3 = out
        out = self.encoder.mod4(out)
        c4 = out
        out = self.encoder.mod5(out)
        out = self.encoder.bn_out(out)
        c5 = out

        x = self.fpn_module([c3, c4, c5])
        if self.upsample_size:
            x = F.upsample(self.classifier(x), mode='bilinear',
                           size=(self.upsample_size[0], self.upsample_size[1]))
        else:
            x = F.upsample(self.classifier(x), mode='bilinear',
                           size=(h // self.mask_downsampling, w // self.mask_downsampling))
        return x

def init_weights(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.nn.init.kaiming_normal(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN) \
                or isinstance(m, InPlaceABNSync):
            nn.init.constant(m.weight, 1.)
            nn.init.constant(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight, .1)
            nn.init.constant(m.bias, 0.)
