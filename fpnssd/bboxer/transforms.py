from fpnssd.albumentations import BasicTransform
import fpnssd.bboxer.functional as F


class BBoxEncoder(BasicTransform):
    def __init__(self, anchors):
        super().__init__(1.0)
        self.anchors = anchors

    def __call__(self, **kwargs):
        bboxes, labels = F.bbox_label_encode(kwargs['bboxes'], kwargs['labels'], self.anchors)
        kwargs.update({'bboxes': bboxes, 'labels': labels})
        return kwargs


class BBoxDecoder(BasicTransform):
    def __init__(self, anchors, score_threshold=0.6, nms_threshold=0.45):
        super().__init__(1.0)
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.anchors = anchors

    def __call__(self, **kwargs):
        bboxes, labels, scores = F.bbox_label_decode(
            bbox_predictions=kwargs['bboxes'],
            label_predictions=kwargs['labels'],
            anchor_bboxes=self.anchors.bboxes,
            nms_threshold=self.nms_threshold,
            score_threshold=self.score_threshold)
        kwargs.update({'bboxes': bboxes, 'labels': labels, 'scores': scores})
        return kwargs

