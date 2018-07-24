import torch


def change_box_order(boxes, order):
    """Change box order between (x_min, y_min, x_max, y_max) and (x_center, y_center, width, height).

    Args:
      boxes: (tensor) bounding boxes, sized [N, 4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N, 4].
    """
    assert order in ['xyxy2xywh', 'xywh2xyxy']
    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a + b)/2, b - a], 1)
    return torch.cat([a - b/2, a + b/2], 1)


def box_clamp(boxes, x_min, y_min, x_max, y_max):
    """Clamp boxes.

    Args:
      boxes: (tensor) bounding boxes of (x_min, y_min, x_max, y_max), sized [N, 4].
      x_min: (number) min value of x.
      y_min: (number) min value of y.
      x_max: (number) max value of x.
      y_max: (number) max value of y.

    Returns:
      (tensor) clamped boxes.
    """
    boxes[:, 0].clamp_(min=x_min, max=x_max)
    boxes[:, 1].clamp_(min=y_min, max=y_max)
    boxes[:, 2].clamp_(min=x_min, max=x_max)
    boxes[:, 3].clamp_(min=y_min, max=y_max)
    return boxes


def box_select(boxes, x_min, y_min, x_max, y_max):
    """Select boxes in range (x_min, y_min, x_max, y_max).

    Args:
      boxes: (tensor) bounding boxes of (x_min, y_min, x_max, y_max), sized [N, 4].
      x_min: (number) min value of x.
      y_min: (number) min value of y.
      x_max: (number) max value of x.
      y_max: (number) max value of y.

    Returns:
      (tensor) selected boxes, sized [M, 4].
      (tensor) selected mask, sized [N, ].
    """
    mask = (boxes[:, 0]>=x_min) & (boxes[:, 1]>=y_min) \
         & (boxes[:, 2]<=x_max) & (boxes[:, 3]<=y_max)
    boxes = boxes[mask, :]
    return boxes, mask


def box_iou(box1, box2):
    """Compute the intersection over union of two set of boxes.

    The box order must be (x_min, y_min, x_max, y_max).

    Args:
      box1: (tensor) bounding boxes, sized [N, 4].
      box2: (tensor) bounding boxes, sized [M, 4].

    Return:
      (tensor) iou, sized [N, M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    # N = box1.size(0)
    # M = box2.size(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)        # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N, ]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M, ]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box_nms(bboxes, scores, threshold=0.5):
    """Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N, 4].
      scores: (tensor) confidence scores, sized [N, ].
      threshold: (float) overlap threshold.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0]
            keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (overlap <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


def meshgrid(x, y, row_major=True):
    """Return meshgrid in range x & y.

    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tensor) meshgrid, sized [x * y, 2]

    Example:
    >> meshgrid(3, 2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3, 2, row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    """
    a = torch.arange(0, x)
    b = torch.arange(0, y)
    xx = a.repeat(y).view(-1, 1)
    yy = b.view(-1, 1).repeat(1, x).view(-1, 1)
    return torch.cat([xx, yy], 1) if row_major else torch.cat([yy, xx], 1)


def tensor2d_argmax(x):
    """Find the max value index(row & col) of a 2D tensor."""
    v, _i = x.max(0)
    _j = v.max(0)[1].item()
    return _i[_j], _j


def bbox_label_encode(bboxes, labels, anchor_bboxes, iou_threshold=0.5):
    """Encode target bounding boxes and labels.

    SSD coding rules:
      tx = (x - anchor_x) / (variance[0] * anchor_w)
      ty = (y - anchor_y) / (variance[0] * anchor_h)
      tw = log(w / anchor_w)
      th = log(h / anchor_h)

    Args:
      bboxes: (tensor) bounding boxes of (x_min, y_min, x_max, y_max), sized [#obj, 4].
      labels: (tensor) object class labels, sized [#obj, ].
      anchor_bboxes: (tensor)
      iou_threshold: (float)

    Returns:
      bboxes: (tensor) encoded bounding boxes, sized [#anchors, 4].
      labels: (tensor) encoded class labels, sized [#anchors, ].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/links/models/ssd/multibox_coder.py
    """
    ious = box_iou(anchor_bboxes, bboxes)  # [#anchors, #obj]
    index = torch.LongTensor(anchor_bboxes.size(0)).fill_(-1)
    masked_ious = ious.clone()
    while True:
        i, j = tensor2d_argmax(masked_ious)
        if masked_ious[i, j] < 1e-6:
            break
        index[i] = j
        masked_ious[i, :] = 0
        masked_ious[:, j] = 0

    mask = (index < 0) & (ious.max(1)[0] >= iou_threshold)
    if mask.any():
        index[mask] = ious[mask].max(1)[1]

    multi_bboxes = bboxes[index.clamp(min=0)]  # negative index not supported
    multi_bboxes = change_box_order(multi_bboxes, 'xyxy2xywh')
    anchor_bboxes = change_box_order(anchor_bboxes, 'xyxy2xywh')

    loc_xy = (multi_bboxes[:, :2] - anchor_bboxes[:, :2]) / anchor_bboxes[:, 2:]
    loc_wh = torch.log(multi_bboxes[:, 2:] / anchor_bboxes[:, 2:])
    multi_bboxes = torch.cat([loc_xy, loc_wh], 1)
    # [0, num_classes - 1] -> [1, num_classes]
    multi_labels = 1 + labels[index.clamp(min=0)]
    # 0 is for background
    multi_labels[index < 0] = 0
    return multi_bboxes, multi_labels


def bbox_label_decode(
        multi_bboxes, multi_labels, anchor_bboxes,
        score_threshold=0.6, nms_threshold=0.45, class_independent_nms=False):
    """Decode predicted loc/cls back to real box locations and class labels.

    Args:
      multi_bboxes: (tensor) predicted loc, sized [#anchors, 4].
      multi_labels: (tensor) predicted conf, sized [#anchors, #classes].
      anchor_bboxes: (tensor)
      score_threshold: (float) threshold for object confidence score.
      nms_threshold: (float) threshold for box nms.
      class_independent_nms: (bool).

    Returns:
      bboxes: (tensor) bbox locations, sized [#obj, 4].
      labels: (tensor) class labels, sized [#obj, ].
    """
    anchor_bboxes = change_box_order(anchor_bboxes, 'xyxy2xywh')
    xy = multi_bboxes[:, :2] * anchor_bboxes[:, 2:] + anchor_bboxes[:, :2]
    wh = multi_bboxes[:, 2:].exp() * anchor_bboxes[:, 2:]
    box_predictions = torch.cat([xy - wh / 2, xy + wh / 2], 1)

    decode = class_independent_decode if class_independent_nms else class_dependent_decode
    bboxes, labels, scores = decode(box_predictions, multi_labels, score_threshold, nms_threshold)
    if bboxes is None:
        return \
            torch.tensor([], dtype=torch.float), \
            torch.tensor([], dtype=torch.long), \
            torch.tensor([], dtype=torch.float)
    else:
        return bboxes, labels, scores


def class_independent_decode(box_predictions, multi_labels, score_threshold, nms_threshold):
    scores, labels = torch.max(multi_labels, dim=0)

    mask = (scores > score_threshold) & (labels > 0)
    bboxes = box_predictions[mask]
    scores = scores[mask]
    labels = labels[mask] - 1
    if len(bboxes):
        keep = box_nms(bboxes, scores, nms_threshold)
        return bboxes[keep], labels[keep], scores[keep]
    else:
        return None, None, None


def class_dependent_decode(box_predictions, multi_labels, score_threshold, nms_threshold):
    bboxes = []
    labels = []
    scores = []
    num_classes = multi_labels.size(0)
    for i in range(num_classes - 1):
        # class i corresponds to (i + 1) column
        score = multi_labels[i + 1]
        mask = score > score_threshold
        if not mask.any():
            continue
        box = box_predictions[mask]
        score = score[mask]

        keep = box_nms(box, score, nms_threshold)
        bboxes.append(box[keep])
        labels.append(torch.empty_like(keep).fill_(i))
        scores.append(score[keep])
    if len(bboxes):
        return torch.cat(bboxes, 0), torch.cat(labels, 0), torch.cat(scores, 0)
    else:
        return None, None, None
