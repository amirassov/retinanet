import yaml

from torch.optim import Adam, SGD
from fpnssd.losses import SSDLoss, FocalLoss
from fpnssd.models import SSD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


OPTIMIZERS = {
    'Adam': Adam,
    'SGD': SGD
}

LOSSES = {
    'SSDLoss': SSDLoss,
    'FocalLoss': FocalLoss
}

SCHEDULERS = {
    'StepLR': StepLR,
    'CosineAnnealingLR': CosineAnnealingLR
}


class SSDConfig:
    def __init__(self, filepath):
        self.filepath = filepath
        self._class2label = None
        self._label2class = None
        self._model = None
        self._config = None
        self._optimizer = None

    @property
    def image_size(self):
        return self.config['bbox_params']['image_size']

    @property
    def config(self):
        if self._config is None:
            with open(self.filepath, 'r') as stream:
                self._config = yaml.load(stream)
        return self._config

    @property
    def train_params(self):
        return self.config['train_params']

    @property
    def data_params(self):
        return self.config['data_params']

    @property
    def class2label(self):
        if self._class2label is None:
            self._class2label = dict(zip(self.config['classes'], range(len(self.config['classes']))))
        return self._class2label

    @property
    def label2class(self):
        return dict(zip(range(len(self.config['classes'])), self.config['classes']))

    @property
    def model(self):
        if self._model is None:
            self._model = SSD(
                label2class=self.label2class,
                bbox_params=self.config['bbox_params'],
                backbone_params=self.config['backbone_params'],
                subnet_params=self.config['subnet_params'])
        return self._model

    @property
    def optimizer(self):
        if self._optimizer is None:
            parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            self._optimizer = OPTIMIZERS[self.train_params['optimizer']](
                parameters,
                **self.train_params['optimizer_params'])
        return self._optimizer

    @property
    def scheduler(self):
        return SCHEDULERS[self.train_params['scheduler']](self.optimizer, **self.train_params['scheduler_params'])

    @property
    def loss(self):
        return LOSSES[self.train_params['loss']](**self.train_params['loss_params'])
