import torch.nn as nn

from .header import MultiBBoxHeader
from .fpn import ResNetFPN
from ..bboxer import BBoxer


class SSD(nn.Module):
    def __init__(self, class2label, bbox_kwargs, feature_extracter_kwargs):
        super().__init__()
        self.class2label = class2label
        self.bboxer = BBoxer(**bbox_kwargs)
        self.feature_extractor = ResNetFPN(**feature_extracter_kwargs)
        self.header = MultiBBoxHeader(num_classes=len(class2label) + 1, num_anchors=self.bboxer.num_anchors)

    def forward(self, x):
        feature_maps = self.feature_extractor(x)
        return self.header(feature_maps)

    def predict(self, x):
        feature_maps = self.feature_extractor(x)
        multi_bboxes, multi_labels = self.header(feature_maps)
        return self.bboxer.decoder(bboxes=multi_bboxes, labels=multi_labels)
