from torch.utils.data import Dataset
import cv2
import numpy as np


class SSDDataset(Dataset):
    """Load image/labels/boxes from a list.

    The sample is like:
    {
        'filepath': 'path',
        'id': 453389,
        'objs': [
            {
                'bottom': 0.36585365853658536,
                'left': 0.10860558712121213,
                'right': 0.21101642377448804,
                'top': 0.7334350027689119
                'class': 0
            },
               ...]
    }
    """
    def __init__(self, samples, box_coder, transform=None):
        """
        Args:
          samples: (list).
          transform: (function) image/box transform.
        """
        self.box_coder = box_coder
        self.transform = transform
        self.filepaths = []
        self.bboxes = []
        self.labels = []
        self._prepare_data(samples)

    def _prepare_data(self, samples):
        for sample in samples:
            self.filepaths.append(sample['filepath'])
            objs = sample['objs']
            box = []
            label = []
            for i, obj in enumerate(objs):
                box.append([obj['left'], obj['bottom'], obj['right'], obj['top']])
                label.append(int(obj['label']))
            self.bboxes.append(np.array(box))
            self.labels.append(label)

    def __getitem__(self, idx):
        """Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        """
        # Load image and boxes.
        filename = self.filepaths[idx]
        image = cv2.imread(filename)
        bboxes = self.bboxes[idx].copy()  # use clone to avoid any potential change.
        labels = self.labels[idx].copy()

        return self.transform(image=image, bboxes=bboxes, labels=labels)

    def __len__(self):
        return len(self.filepaths)
