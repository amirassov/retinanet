import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.nll = nn.NLLLoss(size_average=False)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, *targets):
        """
        loss = SmoothL1Loss(bbox_input, bbox_target) + α * FocalLoss(label_input, label_target).
        """
        bbox_input, label_input = input
        bbox_target, label_target = targets

        positive = label_target > 0
        num_positive = positive.sum().item()

        mask = positive.unsqueeze(2).expand_as(bbox_input)
        bbox_loss = F.smooth_l1_loss(bbox_input[mask], bbox_target[mask], size_average=False)
        label_loss = F.nll_loss((1 - label_input.exp()) ** self.gamma * label_input, label_target, size_average=False)
        loss = (bbox_loss + self.alpha * label_loss) / num_positive
        return loss
