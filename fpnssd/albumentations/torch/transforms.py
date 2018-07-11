import torch
from . import functional as F

from ..core.transforms_interface import BasicTransform


__all__ = ['ToTensor', 'Resize']


class ToTensor(BasicTransform):
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


class Resize(BasicTransform):
    def __init__(self, min_dim=256, max_dim=256):
        super().__init__(1.)
        self.min_dim = min_dim
        self.max_dim = max_dim

    def __call__(self, **kwargs):
        image, scale, left_pad, bottom_pad, right_pad, top_pad = F.resize_image(
            image=kwargs['image'],
            min_dim=self.min_dim,
            max_dim=self.max_dim)

        kwargs.update({'image': image})
        if 'bboxes' in kwargs.keys():
            bboxes = F.resize_bbox(kwargs['bboxes'], scale, left_pad, bottom_pad, right_pad, top_pad)
            kwargs.update({'bboxes': bboxes})
        return kwargs