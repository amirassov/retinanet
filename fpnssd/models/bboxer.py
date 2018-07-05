"""Encode object boxes and labels."""
import math
import torch

from fpnssd.utils import meshgrid
from fpnssd.utils.box import box_iou, box_nms, change_box_order


class BBoxer:
    def __init__(self, image_size, anchor_areas, aspect_ratios, scale_ratios, backbone_strides):
        self.anchor_areas = anchor_areas
        self.aspect_ratios = aspect_ratios
        self.scale_ratios = scale_ratios
        self.backbone_strides = backbone_strides
        self.image_size = torch.FloatTensor(image_size)
        self._num_anchors = None
        self._anchor_boxes = None
        self._anchor_sizes = None

    @property
    def num_anchors(self):
        if self._num_anchors is None:
            self._num_anchors = len(self.aspect_ratios) * len(self.scale_ratios)
        return self._num_anchors

    @property
    def anchor_sizes(self):
        """Compute anchor width and height for each feature map.

        Returns:
          anchor_sizes: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        """
        if self._anchor_sizes is None:
            self._anchor_sizes = []
            for s in self.anchor_areas:
                for ar in self.aspect_ratios:  # w / h = ar
                    h = math.sqrt(s / ar)
                    w = ar * h
                    for sr in self.scale_ratios:  # scale
                        anchor_h = h * sr
                        anchor_w = w * sr
                        self._anchor_sizes.append([anchor_w, anchor_h])
            num_feature_maps = len(self.anchor_areas)
            self._anchor_sizes = torch.FloatTensor(self._anchor_sizes).view(num_feature_maps, -1, 2)
        return self._anchor_sizes

    @property
    def feature_map_sizes(self):
        return [(self.image_size / stride).ceil() for stride in self.backbone_strides]

    @property
    def anchor_boxes(self):
        """Compute anchor boxes for each feature map.

        Returns:
          anchor_boxes: (tensor) anchor boxes for each feature map. Each of size [#anchors, 4],
            where #anchors = fmw * fmh * #anchors_per_cell
        """
        if self._anchor_boxes is None:
            self._anchor_boxes = []
            for feature_map_size, anchor_size in zip(self.feature_map_sizes, self.anchor_sizes):
                grid_size = self.image_size / feature_map_size
                feature_map_w, feature_map_h = int(feature_map_size[0]), int(feature_map_size[1])
                xy = meshgrid(feature_map_w, feature_map_h) + 0.5
                xy = (xy * grid_size).view(feature_map_h, feature_map_w, 1, 2)
                xy = xy.expand(feature_map_h, feature_map_w, self.num_anchors, 2)
                wh = anchor_size.view(1, 1, self.num_anchors, 2)
                wh = wh.expand(feature_map_h, feature_map_w, self.num_anchors, 2)
                box = torch.cat([xy - wh / 2.0, xy + wh / 2.0], 3)
                self._anchor_boxes.append(box.view(-1, 4))
            self._anchor_boxes = torch.cat(self._anchor_boxes, 0)
        return self._anchor_boxes

    def encode(self, boxes, labels):
        """Encode target bounding boxes and class labels.

        SSD coding rules:
          tx = (x - anchor_x) / (variance[0] * anchor_w)
          ty = (y - anchor_y) / (variance[0] * anchor_h)
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (x_min, y_min, x_max, y_max), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj, ].

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors, 4].
          cls_targets: (tensor) encoded class labels, sized [#anchors, ].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/models/ssd/multibox_coder.py
        """

        def argmax(x):
            """Find the max value index(row & col) of a 2D tensor."""
            v, _i = x.max(0)
            _j = v.max(0)[1].item()
            return _i[_j], _j

        anchor_boxes = self.anchor_boxes
        ious = box_iou(anchor_boxes, boxes)  # [#anchors, #obj]
        index = torch.LongTensor(anchor_boxes.size(0)).fill_(-1)
        masked_ious = ious.clone()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i, j] < 1e-6:
                break
            index[i] = j
            masked_ious[i, :] = 0
            masked_ious[:, j] = 0

        mask = (index < 0) & (ious.max(1)[0] >= 0.5)
        if mask.any():
            index[mask] = ious[mask].max(1)[1]

        boxes = boxes[index.clamp(min=0)]  # negative index not supported
        boxes = change_box_order(boxes, 'xyxy2xywh')
        anchor_boxes = change_box_order(anchor_boxes, 'xyxy2xywh')

        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        cls_targets = 1 + labels[index.clamp(min=0)]
        cls_targets[index < 0] = 0
        return loc_targets, cls_targets

    def decode(self, loc_predictions, cls_predictions, score_thresh=0.6, nms_thresh=0.45):
        """Decode predicted loc/cls back to real box locations and class labels.

        Args:
          loc_predictions: (tensor) predicted loc, sized [#anchors, 4].
          cls_predictions: (tensor) predicted conf, sized [#anchors, #classes].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.

        Returns:
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj, ].
        """
        anchor_boxes = change_box_order(self.anchor_boxes, 'xyxy2xywh')
        xy = loc_predictions[:, :2] * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = loc_predictions[:, 2:].exp() * anchor_boxes[:, 2:]
        box_predictions = torch.cat([xy - wh / 2, xy + wh / 2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_predictions.size(1)
        for i in range(num_classes - 1):
            score = cls_predictions[:, i + 1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_predictions[mask]
            score = score[mask]

            keep = box_nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.empty_like(keep).fill_(i))
            scores.append(score[keep])

        boxes = torch.cat(boxes, 0)
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)
        return boxes, labels, scores


def test():
    box_coder = BBoxer(
        image_size=(128, 128),
        anchor_areas=(4, 16),
        aspect_ratios=(1, 2),
        scale_ratios=(1, 2),
        backbone_strides=(4, 8))
    print(box_coder.anchor_boxes.size())
    boxes = torch.FloatTensor([[0, 0, 100, 100], [100, 100, 200, 200]])
    labels = torch.LongTensor([0, 1])
    loc_targets, cls_targets = box_coder.encode(boxes, labels)
    print(loc_targets.size(), cls_targets.size())
