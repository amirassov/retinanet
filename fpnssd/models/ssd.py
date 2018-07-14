import torch
import torch.nn as nn

from .subnet import Subnet
from .fpn import ResNetFPN
from ..bboxer import BBoxer
import torch.nn.functional as F


class SSD(nn.Module):
    def __init__(self, label2class, bbox_params, backbone_params, subnet_params):
        super().__init__()
        self.label2class = label2class
        self.bboxer = BBoxer(**bbox_params)
        self.backbone = ResNetFPN(**backbone_params)
        self.label_subnet = Subnet(
            num_classes=len(self.label2class) + 1,
            num_anchors=self.bboxer.num_anchors,
            **subnet_params)
        self.bbox_subnet = Subnet(
            num_classes=4,
            num_anchors=self.bboxer.num_anchors,
            **subnet_params)

    def forward(self, x):
        feature_maps = self.backbone(x)
        multi_bboxes = self.bbox_subnet(feature_maps)
        multi_labels = self.label_subnet(feature_maps)
        multi_labels = F.log_softmax(multi_labels, dim=2)
        multi_labels = multi_labels.permute(0, 2, 1)
        return multi_bboxes, multi_labels

    def cuda(self, device=None):
        self.bboxer.cuda(device=device)
        return super().cuda(device=device)

    def cpu(self):
        self.bboxer.cpu()
        return super().cpu()

    def predict(self, x):
        multi_bboxes, multi_labels = self.forward(x)
        multi_labels = multi_labels.exp()
        bboxes = []
        classes = []
        scores = []
        for multi_bbox, multi_label in zip(multi_bboxes, multi_labels):
            _bboxes, _labels, _scores = self.bboxer.decode(multi_bboxes=multi_bbox, multi_labels=multi_label)
            bboxes.append(_bboxes)
            classes.append(_labels)
            scores.append(_scores)
        return torch.cat(bboxes, dim=0), torch.cat(classes, dim=0), torch.cat(scores, dim=0)
