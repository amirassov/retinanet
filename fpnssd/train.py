import os
from collections import defaultdict
from copy import deepcopy

import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam, SGD
from tqdm import tqdm

from fpnssd.losses import SSDLoss, FocalLoss
from datetime import datetime


optimizers = {
    'Adam': Adam,
    'SGD': SGD
}

losses = {
    'SSDLoss': SSDLoss,
    'FocalLoss': FocalLoss
}


def adjust_lr(epoch, init_lr=3e-4, num_epochs_per_decay=100, lr_decay_factor=0.3):
    lr = init_lr * (lr_decay_factor ** (epoch // num_epochs_per_decay))
    return lr



class PytorchTrain:
    def __init__(
        self, model, epochs, loss,
        model_dir, log_dir, metrics, loss_args, optimizer, optimizer_args
    ):
        self.model = model.cuda()
        self.epochs = epochs

        self.log_dir = os.path.join(log_dir, self.model_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model_dir = os.path.join(model_dir, self.model_name)
        os.makedirs(self.model_dir, exist_ok=True)

        self.lr = optimizer_args['lr']
        self.optimizer = optimizers[optimizer](self.model.parameters(), **optimizer_args)

        self.criterion = losses[loss](**loss_args).cuda()
        self.writer = SummaryWriter(self.log_dir)
        self.metrics = metrics

    @property
    def model_name(self):
        return f"{self.model.__class__}_{datetime.now()}".replace(" ", "")

    def _run_one_epoch(self, epoch, loader, is_train=True, lr=None):
        epoch_report = defaultdict(float)
        grad_manager = torch.enable_grad() if is_train else torch.no_grad()
        if is_train:
            progress_bar = tqdm(
                enumerate(loader), total=len(loader),
                desc="Epoch {}, lr {}".format(epoch, lr), ncols=0
            )
        else:
            progress_bar = enumerate(loader)
        with grad_manager:
            for i, data in progress_bar:
                step_report = self._make_step(data, is_train)

                for key, value in step_report.items():
                    epoch_report[key] += value

                if is_train:
                    progress_bar.set_postfix(
                        **{key: "{:.5f}".format(value.cpu().numpy() / (i + 1)) for key, value in epoch_report.items()}
                    )

        return {key: value.cpu().numpy() / len(loader) for key, value in epoch_report.items()}

    def _make_step(self, data, is_train):
        report = {}

        images = data['image'].cuda()
        boxes = data['boxes'].cuda()
        labels = data['labels'].cuda()

        if is_train:
            self.optimizer.zero_grad()

        box_predictions, label_predictions = self.model(images)
        loss = self.criterion(box_predictions, boxes, label_predictions, labels)
        report['losses'] = loss.data

        for name, func in self.metrics:
            report[name] = func(box_predictions, boxes, label_predictions, labels).data

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return report

    def fit(self, train_loader, val_loader):
        best_epoch = -1
        best_loss = float('inf')
        try:
            for epoch in range(self.epochs):
                lr = adjust_lr(epoch, self.lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                self.model.train()
                train_metrics = self._run_one_epoch(epoch, train_loader, lr=lr)

                self.model.eval()
                val_metrics = self._run_one_epoch(epoch, val_loader, is_train=False)

                print(" | ".join("{}: {:.5f}".format(key, float(value)) for key, value in val_metrics.items()))

                for key, value in train_metrics.items():
                    self.writer.add_scalar('train/{}'.format(key), float(value), global_step=epoch)

                for key, value in val_metrics.items():
                    self.writer.add_scalar('val/{}'.format(key), float(value), global_step=epoch)

                loss = float(val_metrics['losses'])

                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    torch.save(deepcopy(self.model), os.path.join(self.model_dir, 'best.pth'))

        except KeyboardInterrupt:
            print('done.')

        self.writer.close()
        print('Finished: best losses {:.5f} on epoch {}'.format(best_loss, best_epoch))
