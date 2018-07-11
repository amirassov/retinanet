import numpy as np
import torch
import torchvision.transforms.functional as F
import cv2


def img_to_tensor(im, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(im / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor


def mask_to_tensor(mask, num_classes, sigmoid):
    if num_classes > 1:
        if not sigmoid:
            # softmax
            long_mask = np.zeros((mask.shape[:2]), dtype=np.int64)
            if len(mask.shape) == 3:
                for c in range(mask.shape[2]):
                    long_mask[mask[..., c] > 0] = c
            else:
                long_mask[mask > 127] = 1
                long_mask[mask == 0] = 0
            mask = long_mask
        else:
            mask = np.moveaxis(mask / (255. if mask.dtype == np.uint8 else 1), -1, 0).astype(np.float32)
    else:
        mask = np.expand_dims(mask / (255. if mask.dtype == np.uint8 else 1), 0).astype(np.float32)
    return torch.from_numpy(mask)


def resize_image(image, min_dim=256, max_dim=256):
    image_dtype = image.dtype
    h, w = image.shape[:2]
    scale = max(1, min_dim / min(h, w))

    image_max = max(h, w)
    if round(image_max * scale) > max_dim:
        scale = max_dim / image_max

    if scale != 1:
        image = cv2.resize(image, (round(w * scale), round(h * scale)))

    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    return image.astype(image_dtype), scale, left_pad, bottom_pad, right_pad, top_pad


def resize_bbox(bboxes, scale, left_pad, bottom_pad, right_pad, top_pad):
    bboxes *= scale
    bboxes[:, 0] += left_pad
    bboxes[:, 1] += bottom_pad
    bboxes[:, 2] += right_pad
    bboxes[:, 3] += top_pad
    return bboxes