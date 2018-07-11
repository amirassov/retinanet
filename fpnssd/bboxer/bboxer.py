import math
import torch
from . import functional as F
from .transforms import BBoxDecoder, BBoxEncoder


class BBoxer:
    def __init__(
            self, image_size, areas, aspect_ratios, scale_ratios,
            backbone_strides, iou_threshold, score_threshold, nms_threshold):
        self.areas = areas
        self.aspect_ratios = aspect_ratios
        self.scale_ratios = scale_ratios
        self.backbone_strides = backbone_strides
        self.image_size = torch.FloatTensor(image_size)
        self._num_anchors = None
        self._bboxes = None
        self._sizes = None
        self.encoder = BBoxEncoder(
            anchor_bboxes=self.bboxes,
            iou_threshold=iou_threshold)
        self.decoder = BBoxDecoder(
            anchor_bboxes=self.bboxes,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold)

    @property
    def num_anchors(self):
        if self._num_anchors is None:
            self._num_anchors = len(self.aspect_ratios) * len(self.scale_ratios)
        return self._num_anchors

    @property
    def sizes(self):
        """Compute anchor width and height for each feature map.

        Returns:
          anchor_sizes: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        """
        if self._sizes is None:
            self._sizes = []
            for s in self.areas:
                for ar in self.aspect_ratios:  # w / h = ar
                    h = math.sqrt(s / ar)
                    w = ar * h
                    for sr in self.scale_ratios:  # scale
                        anchor_h = h * sr
                        anchor_w = w * sr
                        self._sizes.append([anchor_w, anchor_h])
            self._sizes = torch.FloatTensor(self._sizes).view(len(self.areas), -1, 2)
        return self._sizes

    @property
    def feature_map_sizes(self):
        return [(self.image_size / stride).ceil() for stride in self.backbone_strides]

    @property
    def bboxes(self):
        """Compute anchor bboxes for each feature map.

        Returns:
          anchor_bboxes: (tensor) anchor boxes for each feature map. Each of size [#anchors, 4],
            where #anchors = fmw * fmh * #anchors_per_cell
        """
        if self._bboxes is None:
            self._bboxes = []
            for feature_map_size, anchor_size in zip(self.feature_map_sizes, self.sizes):
                grid_size = self.image_size / feature_map_size
                feature_map_w, feature_map_h = int(feature_map_size[0]), int(feature_map_size[1])
                xy = F.meshgrid(feature_map_w, feature_map_h) + 0.5
                xy = (xy * grid_size).view(feature_map_h, feature_map_w, 1, 2)
                xy = xy.expand(feature_map_h, feature_map_w, self.num_anchors, 2)
                wh = anchor_size.view(1, 1, self.num_anchors, 2)
                wh = wh.expand(feature_map_h, feature_map_w, self.num_anchors, 2)
                box = torch.cat([xy - wh / 2.0, xy + wh / 2.0], 3)
                self._bboxes.append(box.view(-1, 4))
            self._bboxes = torch.cat(self._bboxes, 0)
        return self._bboxes
