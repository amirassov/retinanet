import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def _label_loss(self, label_input, label_target):
        batch_size, classes_no, anchors_no = label_input.size()

        label_target = label_target.view(batch_size * anchors_no)
        permute = label_input.permute(0, 2, 1).contiguous()
        label_input = permute.view(batch_size * anchors_no, classes_no)

        # Filter anchors with -1 label from loss computation
        not_ignored = label_target >= 0
        label_target = label_target[not_ignored]
        label_input = label_input[not_ignored]

        return F.nll_loss(
            (1 - label_input.exp()) ** self.gamma * label_input,
            label_target,
            reduction='sum')

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

        loss = (bbox_loss + self.alpha * label_loss) / (num_positive + 1) / (1 + self.alpha)
        return loss
