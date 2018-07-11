import torch
import torch.nn as nn

import torch.nn.functional as F


class MultiBBoxHeader(nn.Module):
    def __init__(self, num_classes, num_anchors):
        self.n_layers = 4

        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.bbox_header = self._make_header(self.num_anchors * 4)
        self.label_header = self._make_header(self.num_anchors * self.num_classes)

    def _make_header(self, out_planes):
        layers = []
        for _ in range(self.n_layers):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        multi_bboxes = []
        multi_labels = []
        for feature_map in x:
            multi_bbox = self.bbox_header(feature_map)
            multi_label = self.label_header(feature_map)
            # [batch_size, num_anchors * 4, h, w] -> [batch_size, h, w, num_anchors * 4] -> [batch_size, h * w * num_anchors, 4]
            multi_bbox = multi_bbox.permute(0, 2, 3, 1).reshape(multi_bbox.size(0), -1, 4)
            multi_label = multi_label.permute(0, 2, 3, 1).reshape(multi_label.size(0), -1, self.num_classes)
            multi_bboxes.append(multi_bbox)
            multi_labels.append(multi_label)
        multi_bboxes = torch.cat(multi_bboxes, 1)
        multi_labels = torch.cat(multi_labels, 1)

        multi_labels = F.log_softmax(multi_labels, dim=2)
        multi_labels = multi_labels.permute(0, 2, 1)
        return multi_bboxes, multi_labels
