import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(pred,
                       target,
                       gamma=2.0,
                       alpha=0.25):
    """
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target))
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * weight
    return loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def _label_loss(self, label_input, label_target):
        batch_size, num_anchors, num_classes = label_input.size()
        loss = 0
        label_target = label_target.view(batch_size * num_anchors)
        label_input = label_input.view(batch_size * num_anchors, num_classes)
        for cls in range(num_classes):
            cls_label_target = (label_target == (cls + 1)).long()
            cls_label_input = label_input[..., cls]

            # Filter anchors with -1 label from loss computation
            not_ignored = cls_label_target >= 0
            cls_label_target = cls_label_target[not_ignored]
            cls_label_input = cls_label_input[not_ignored]

            loss += sigmoid_focal_loss(cls_label_input, cls_label_target, gamma=self.gamma, alpha=self.alpha)
        return loss

    @staticmethod
    def _bbox_loss(bbox_input, bbox_target, positive):
        mask = positive.unsqueeze(2).expand_as(bbox_input)
        return F.smooth_l1_loss(
            bbox_input[mask],
            bbox_target[mask],
            reduction='sum')

    def forward(self, input, *targets):
        """
        loss = SmoothL1Loss(bbox_input, bbox_target) + α * FocalLoss(label_input, label_target).

        Ignores anchors having -1 target label
        """
        bbox_input, label_input = input
        bbox_target, label_target = targets

        positive = label_target > 0
        num_positive = positive.sum().item()

        label_loss = self._label_loss(label_input, label_target)
        bbox_loss = self._bbox_loss(bbox_input, bbox_target, positive)
        loss = (bbox_loss + label_loss) / (num_positive + 1)
        return loss
