import torch
from . import functional as F

from ..core.transforms_interface import BasicTransform


__all__ = ['ToTensor']


class ToTensor(BasicTransform):
    def apply(self, img, **params):
        pass

    def __init__(self, num_classes=1, sigmoid=True, normalize=None):
        super().__init__(1.)
        self.num_classes = num_classes
        self.sigmoid = sigmoid
        self.normalize = normalize

    def __call__(self, **kwargs):
        kwargs.update({'image': F.img_to_tensor(kwargs['image'], self.normalize)})
        if 'bboxes' in kwargs.keys():
            kwargs.update({'bboxes': torch.FloatTensor(kwargs['bboxes'])})
        if 'mask' in kwargs.keys():
            kwargs.update({'mask': F.mask_to_tensor(kwargs['mask'], self.num_classes, sigmoid=self.sigmoid)})
        if 'labels' in kwargs.keys():
            kwargs.update({'labels': torch.LongTensor(kwargs['labels'])})
        return kwargs
