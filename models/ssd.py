from typing import Dict

import torch.nn as nn

from .subnet import Subnet
from .fpn import RetinaNetFPN
from ..bboxer import BBoxer
import torch.nn.functional as F


class SSD(nn.Module):
    def __init__(self, classes, bbox_params, backbone_params: Dict, subnet_params):
        super().__init__()
        self.label2class = dict(zip(range(len(classes)), classes))
        self.bboxer = BBoxer(**bbox_params)
        self.backbone = RetinaNetFPN(**backbone_params)

        self.label_subnet = Subnet(
            num_classes=len(self.label2class) + 1,
            num_anchors=self.bboxer.num_anchors,
            **subnet_params)
        self.bbox_subnet = Subnet(
            num_classes=4,
            num_anchors=self.bboxer.num_anchors,
            **subnet_params)

    def forward(self, x, inference=False):
        feature_maps = self.backbone(x)
        multi_bboxes = self.bbox_subnet(feature_maps)
        multi_labels = self.label_subnet(feature_maps)
        multi_labels = F.log_softmax(multi_labels, dim=2)
        multi_labels = multi_labels.permute(0, 2, 1)
        if inference:
            return zip(*[self.bboxer.decode(y, z) for y, z in zip(multi_bboxes, multi_labels.exp())])
        else:
            return multi_bboxes, multi_labels

    def cuda(self, device=None):
        self.bboxer.cuda(device=device)
        return super().cuda(device=device)

    def to(self, device=None):
        self.bboxer.to(device=device)
        return super().to(device=device)

    def cpu(self):
        self.bboxer.cpu()
        return super().cpu()
