# Based on https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
import colorsys
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
    plt.show()


def random_colors(n, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, bboxes, classes=None, scores=None, title="", figsize=(16, 16), ax=None):
    if not len(bboxes):
        print("\n*** No instances to display *** \n")

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = random_colors(len(bboxes))

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    for i, bbox in enumerate(bboxes):
        color = colors[i]
        # Bounding box
        if not np.any(bboxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        left, bottom, right, top = bboxes[i]
        p = patches.Rectangle(
            xy=(left, bottom),
            width=right - left,
            height=top - bottom,
            linewidth=2,
            alpha=0.7,
            linestyle="dashed",
            edgecolor=color,
            facecolor='none')
        ax.add_patch(p)

        # Label
        cls = classes[i] if classes is not None else ''
        caption = "{}".format(cls)
        ax.text(left, top + 3, caption, color='b', size=15, backgroundcolor="none")

        # Score
        score = scores[i] if scores is not None else None
        caption = "{:.1f}".format(score) if score else ''
        ax.text(left, bottom - 3, caption, color='b', size=15, backgroundcolor="none")
    ax.imshow(image.astype(np.uint8))
    if auto_show:
        plt.show()
