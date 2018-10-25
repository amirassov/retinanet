import torch
import torch.nn as nn


class Subnet(nn.Module):
    def __init__(self, num_classes, num_anchors, num_layers):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_layers = num_layers
        self.layers = self._make_layers(self.num_anchors * self.num_classes)

    def _make_layers(self, out_planes):
        layers = []
        for _ in range(self.num_layers):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        for feature_map in x:
            output = self.layers(feature_map)
            # [batch_size, num_anchors * num_classes, h, w] -> [batch_size, h, w, num_anchors * num_classes]
            output = output.permute(0, 2, 3, 1)
            # [batch_size, h, w, num_anchors * num_classes] -> [batch_size, h * w * num_anchors, num_classes]
            output = output.reshape(output.size(0), -1, self.num_classes)
            outputs.append(output)
        outputs = torch.cat(outputs, 1)
        return outputs
