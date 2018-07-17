import os
from collections import defaultdict
from .callbacks import Callbacks
import torch
from tqdm import tqdm
from .parallel import DataParallelCriterion, DataParallelModel


class MetricsCollection:
    def __init__(self):
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}


class Runner:
    def __init__(self, model, epochs, loss, name, model_dir, optimizer, scheduler, callbacks=None):
        self.model = DataParallelModel(model).cuda()
        self.epochs = epochs
        self.name = name
        self.model_dir = os.path.join(model_dir, self.name)
        os.makedirs(self.model_dir, exist_ok=True)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = DataParallelCriterion(loss).cuda()
        self.metrics_collection = MetricsCollection()
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_trainer(self)

    def _run_one_epoch(self, epoch, loader, is_train=True):
        epoch_report = defaultdict(float)
        grad_manager = torch.enable_grad() if is_train else torch.no_grad()
        if is_train:
            progress_bar = tqdm(
                enumerate(loader), total=len(loader),
                desc="Epoch {}".format(epoch), ncols=0
            )
        else:
            progress_bar = enumerate(loader)
        with grad_manager:
            for i, data in progress_bar:
                self.callbacks.on_batch_begin(i)
                step_report = self._make_step(data, is_train)

                for key, value in step_report.items():
                    epoch_report[key] += value

                if is_train:
                    progress_bar.set_postfix(**{k: "{:.5f}".format(v.item() / (i + 1)) for k, v in epoch_report.items()})

                self.callbacks.on_batch_end(i)
        return {key: value.cpu().numpy() / len(loader) for key, value in epoch_report.items()}

    def _make_step(self, data, is_train):
        report = {}

        images = data['image'].cuda()
        multi_bboxes = data['bboxes'].cuda()
        multi_labels = data['labels'].cuda()
        if is_train:
            self.optimizer.zero_grad()

        predictions = self.model(images)
        loss = self.criterion(predictions, multi_bboxes, multi_labels)
        report['loss'] = loss.data

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return report

    def fit(self, train_loader, val_loader):
        try:
            self.callbacks.on_train_begin()
            for epoch in range(self.epochs):
                self.callbacks.on_epoch_begin(epoch)

                if self.scheduler is not None:
                    self.scheduler.step()

                for param_group in self.optimizer.param_groups:
                    print(param_group['lr'])

                self.model.train()
                self.metrics_collection.train_metrics = self._run_one_epoch(epoch, train_loader)

                self.model.eval()
                self.metrics_collection.val_metrics = self._run_one_epoch(epoch, val_loader, is_train=False)

                print(" | ".join("{}: {:.5f}".format(k, v.item()) for k, v in self.metrics_collection.val_metrics.items()))
                self.callbacks.on_epoch_end(epoch)
            self.callbacks.on_train_end()
        except KeyboardInterrupt:
            print('done.')
