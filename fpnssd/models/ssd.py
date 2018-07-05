import torch
import torch.nn as nn

from fpnssd.models.fpn import ResNetFPN
import torch.nn.functional as F
from fpnssd.models import BBoxer


ENCODERS = {
    'ResNetFPN': ResNetFPN
}


def _make_head(out_planes):
    layers = []
    for _ in range(4):
        layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(True))
    layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
    return nn.Sequential(*layers)


class FPNSSD(nn.Module):
    def __init__(self, is_train, num_classes, encoder, encoder_args, bboxer_args):
        super().__init__()
        self.is_train = is_train
        self.bboxer = BBoxer(**bboxer_args)
        self.fpn = ENCODERS[encoder](**encoder_args)
        self.num_classes = num_classes
        self.loc_head = _make_head(self.bboxer.num_anchors * 4)
        self.cls_head = _make_head(self.bboxer.num_anchors * self.num_classes)

    def forward(self, x, boxes=None, labels=None):
        loc_predictions = []
        cls_predictions = []
        feature_maps = self.fpn(x)
        for feature_map in feature_maps:
            loc_prediction = self.loc_head(feature_map)
            cls_prediction = self.cls_head(feature_map)
            loc_prediction = loc_prediction.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
            cls_prediction = cls_prediction.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)
            loc_predictions.append(loc_prediction)
            cls_predictions.append(cls_prediction)
        loc_predictions = torch.cat(loc_predictions, 1)
        cls_predictions = torch.cat(cls_predictions, 1)

        cls_predictions = F.log_softmax(cls_predictions, dim=2)
        cls_predictions = cls_predictions.permute(0, 2, 1)

        if self.is_train:
            loc_targets, cls_targets = self.bboxer.encode(boxes=boxes, labels=labels)
            return loc_predictions, cls_predictions, loc_targets, cls_targets
        else:
            loc_predictions, cls_predictions = self.bboxer.decode(
                loc_predictions=loc_predictions,
                cls_predictions=cls_predictions)
            return loc_predictions, cls_predictions


# def test():
#     net = FPNSSD(21, 9, 'ResNetFPN', {'n_layer': 50})
#     loc_predictions, cls_predictions = net(Variable(torch.randn(1, 3, 128, 128)))
#     print(loc_predictions.size(), cls_predictions.size())
