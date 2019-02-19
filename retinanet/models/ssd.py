from typing import Dict

import torch.nn as nn

from retinanet.models.subnet import Subnet
from retinanet.models.fpn import RetinaNetFPN
from retinanet.bboxer import BBoxer


class SSD(nn.Module):
    def __init__(self, classes, bbox_params, fpn_params: Dict, subnet_params):
        super().__init__()
        self.label2class = dict(zip(range(len(classes)), classes))
        self.num_classes = len(self.label2class)
        self.bboxer = BBoxer(**bbox_params)
        self.backbone = RetinaNetFPN(**fpn_params)

        self.label_subnet = Subnet(
            num_classes=self.num_classes,
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
        if inference:
            return zip(*[self.bboxer.decode(y, z) for y, z in zip(multi_bboxes, multi_labels.sigmoid())])
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


if __name__ == '__main__':
    params = {
      "classes": ["object"],
      "bbox_params": {
        "image_size": [1024, 1024],
        "areas": [1000, 2000, 3000, 5000],
        "aspect_ratios": [0.3, 1.5, 2.0],
        "scale_ratios": [1.0, 1.2],
        "backbone_strides": [4, 8, 16, 32],
        "iou_threshold": 0.5,
        "score_threshold": 0.5,
        "nms_threshold": 0.3,
        "ignore_threshold": 0.4,
        "class_independent_nms": True
      },
      "fpn_params": {
          "backbone_path": "retinanet.models.fpn.ResNetBackbone",
          "backbone_params": {
            "pretrained": True,
            "architecture": "resnet18"}
      },
      "subnet_params": {
        "num_layers": 4
      }
    }
    ssd = SSD(**params)
    import torch
    tensor = torch.randn([2, 3, 224, 224], dtype=torch.float)
    output = ssd(tensor)
