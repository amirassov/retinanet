import colorsys
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib import patches
from matplotlib.patches import Polygon
from skimage.measure import find_contours


def vis_image(img, boxes=None, label_names=None, scores=None):
    """Visualize a color image.

    Args:
      img: (PIL.Image/tensor) image to visualize.
      boxes: (tensor) bounding boxes, sized [#obj, 4].
      label_names: (list) label names.
      scores: (list) confidence scores.

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/visualization/vis_bbox.py
      https://github.com/chainer/chainercv/blob/master/chainercv/visualization/vis_image.py
    """
    # Plot image
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if isinstance(img, torch.Tensor):
        img = torchvision.transforms.ToPILImage()(img)
    ax.imshow(img)

    # Plot boxes
    if boxes is not None:
        for i, bb in enumerate(boxes):
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0] + 1
            height = bb[3] - bb[1] + 1

            ax.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor='red', linewidth=2))

            caption = []
            if label_names is not None:
                caption.append(label_names[i])

            if scores is not None:
                caption.append('{:.2f}'.format(scores[i]))

            if len(caption) > 0:
                ax.text(bb[0], bb[1],
                        ': '.join(caption),
                        style='italic',
                        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    # Show
    plt.show()


def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        x1, y1, x2, y2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            label = class_names[class_id]
            caption = "{}".format(label)
        else:
            caption = captions[i]
        ax.text(x1, y2 + 1, caption,
                color='b', size=11, backgroundcolor="none")

        # Score
        if not captions:
            score = scores[i] if scores is not None else None
            caption = "{:.1f}".format(score) if score else ''
        else:
            caption = captions[i]
        ax.text(x1, y1 - 1, caption,
                color='b', size=11, backgroundcolor="none")
    ax.imshow(image.astype(np.uint8))
    if auto_show:
        plt.show()
