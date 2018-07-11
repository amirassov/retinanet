import yaml

from torch.optim import Adam, SGD
from fpnssd.losses import SSDLoss, FocalLoss


OPTIMIZERS = {
    'Adam': Adam,
    'SGD': SGD
}

LOSSES = {
    'SSDLoss': SSDLoss,
    'FocalLoss': FocalLoss
}


def read_config(filepath):
    with open(filepath, 'r') as stream:
        config = yaml.load(stream)
    config['class2label'] = dict(zip(config['classes'], range(len(config['classes']))))
    return config
