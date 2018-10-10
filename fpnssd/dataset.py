from torch.utils.data import Dataset
import cv2
import numpy as np
import os


class SSDDataset(Dataset):
    """Load image/labels/boxes from a list.

    The sample is like:
    {
        'name': 'name.jpg'
        'objs': [
            {
                'bottom': 0.36585365853658536,
                'left': 0.10860558712121213,
                'right': 0.21101642377448804,
                'top': 0.7334350027689119
                'class': 'class_name'
            },
               ...]
    }
    """
    def __init__(self, class2label, samples, transform=None, data_dir=None):
        """
        Args:
          samples: (list).
          transform: (function) image/box transform.
        """
        self.data_dir = data_dir
        self.class2label = class2label
        self.transform = transform
        self.filepaths = []
        self.bboxes = []
        self.labels = []
        self._prepare_data(samples)

    def _prepare_data(self, samples):

        for sample in samples:
            self.filepaths.append(os.path.join(self.data_dir, sample['name']))

            objs = sample['objs']
            bbox = []
            label = []
            for i, obj in enumerate(objs):
                bbox.append([obj['left'], obj['top'], obj['right'], obj['bottom']])
                label.append(int(self.class2label[obj['class']]))
            self.bboxes.append(np.array(bbox))
            self.labels.append(label)

    def __getitem__(self, i):
        """Load image.

        Args:
          i: (int) image index.

        Returns:
          img: (tensor) image tensor.
          bboxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        """
        # Load image and boxes.
        filename = self.filepaths[i]
        image = cv2.imread(filename)
        bboxes = self.bboxes[i].copy()  # use clone to avoid any potential change.
        labels = self.labels[i].copy()

        return self.transform(image=image, bboxes=bboxes, labels=labels)

    def __len__(self):
        return len(self.filepaths)
