import math
import torch
import numpy as np
from . import functional as F


class BBoxer:
    def __init__(
            self, image_size, areas, aspect_ratios, scale_ratios,
            backbone_strides, iou_threshold, score_threshold, nms_threshold, class_independent_nms):
        self.class_independent_nms = class_independent_nms
        self.areas = areas
        self.aspect_ratios = aspect_ratios
        self.scale_ratios = scale_ratios
        self.backbone_strides = backbone_strides
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.image_size = torch.tensor(image_size, dtype=torch.float)
        self._num_anchors = None
        self._anchor_bboxes = None
        self._sizes = None

    def cuda(self, device=None):
        self._anchor_bboxes = self.anchor_bboxes.cuda(device=device)
        return self

    def to(self, device=None):
        self._anchor_bboxes = self.anchor_bboxes.to(device=device)
        return self

    def cpu(self):
        self._anchor_bboxes = self.anchor_bboxes.cpu()
        return self

    @property
    def num_anchors(self):
        if self._num_anchors is None:
            self._num_anchors = len(self.aspect_ratios) * len(self.scale_ratios)
        return self._num_anchors

    @property
    def sizes(self):
        if self._sizes is None:
            self._sizes = []
            for s in self.areas:
                for ar in self.aspect_ratios:
                    h = math.sqrt(s / ar)
                    w = ar * h
                    for sr in self.scale_ratios:
                        anchor_h = h * sr
                        anchor_w = w * sr
                        self._sizes.append([anchor_w, anchor_h])
            self._sizes = torch.tensor(self._sizes, dtype=torch.float).view(len(self.areas), -1, 2)
        return self._sizes

    @property
    def feature_map_sizes(self):
        return [(self.image_size / stride).ceil() for stride in self.backbone_strides]

    @property
    def anchor_bboxes(self):
        if self._anchor_bboxes is None:
            self._anchor_bboxes = []
            for feature_map_size, anchor_size in zip(self.feature_map_sizes, self.sizes):
                grid_size = self.image_size / feature_map_size
                feature_map_h, feature_map_w = int(feature_map_size[0]), int(feature_map_size[1])
                xy = F.meshgrid(feature_map_w, feature_map_h) + 0.5
                xy = (xy * grid_size).view(feature_map_h, feature_map_w, 1, 2)
                xy = xy.expand(feature_map_h, feature_map_w, self.num_anchors, 2)
                wh = anchor_size.view(1, 1, self.num_anchors, 2)
                wh = wh.expand(feature_map_h, feature_map_w, self.num_anchors, 2)
                box = torch.cat([xy - wh / 2.0, xy + wh / 2.0], 3)
                self._anchor_bboxes.append(box.view(-1, 4))
            self._anchor_bboxes = torch.cat(self._anchor_bboxes, 0)
        return self._anchor_bboxes

    def encode(self, bboxes, labels):
        return F.bbox_label_encode(
            bboxes=bboxes,
            labels=labels,
            anchor_bboxes=self.anchor_bboxes,
            iou_threshold=self.iou_threshold)

    def decode(self, multi_bboxes, multi_labels):
        return F.bbox_label_decode(
            multi_bboxes=multi_bboxes,
            multi_labels=multi_labels,
            anchor_bboxes=self.anchor_bboxes,
            nms_threshold=self.nms_threshold,
            score_threshold=self.score_threshold,
            class_independent_nms=self.class_independent_nms)


class BBoxTransform(object):
    def __init__(self, transform, bboxer, p=1.):
        self.transform = transform
        self.bboxer = bboxer
        self.p = p

    def __call__(self, **data):
        if np.random.random() < self.p:
            data = self.transform(**data)
        multi_bboxes, multi_labels = self.bboxer.encode(bboxes=data['bboxes'], labels=data['labels'])
        data.update({'bboxes': multi_bboxes, 'labels': multi_labels})
        return data
