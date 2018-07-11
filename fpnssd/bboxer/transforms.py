from ..albumentations import BasicTransform
from . import functional as F


class BBoxEncoder(BasicTransform):
    def __init__(self, anchor_bboxes, iou_threshold):
        super().__init__(1.0)
        self.anchor_bboxes = anchor_bboxes
        self.iou_threshold = iou_threshold

    def __call__(self, **kwargs):
        bboxes, labels = F.bbox_label_encode(
            bboxes=kwargs['bboxes'],
            labels=kwargs['labels'],
            anchor_bboxes=self.anchor_bboxes,
            iou_threshold=self.iou_threshold)
        kwargs.update({'bboxes': bboxes, 'labels': labels})
        return kwargs


class BBoxDecoder(BasicTransform):
    def __init__(self, anchor_bboxes, score_threshold=0.6, nms_threshold=0.45):
        super().__init__(1.0)
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.anchor_bboxes = anchor_bboxes

    def __call__(self, **kwargs):
        bboxes, labels, scores = F.bbox_label_decode(
            multi_bboxes=kwargs['bboxes'],
            multi_labels=kwargs['labels'],
            anchor_bboxes=self.anchor_bboxes,
            nms_threshold=self.nms_threshold,
            score_threshold=self.score_threshold)
        kwargs.update({'bboxes': bboxes, 'labels': labels, 'scores': scores})
        return kwargs
