import os
from collections import defaultdict
from .callbacks import Callbacks
import torch
from tqdm import tqdm
from .parallel import DataParallelCriterion, DataParallelModel


class Metrics:
    def __init__(self):
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}


class Runner:
    def __init__(self, model, epochs, loss, name, model_dir, optimizer, scheduler, batch_handler, callbacks=None):
        self.model = DataParallelModel(model).cuda()
        self.epochs = epochs
        self.name = name
        self.model_dir = os.path.join(model_dir, self.name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.batch_handler = batch_handler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = DataParallelCriterion(loss).cuda()
        self.metrics = Metrics()
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_trainer(self)

    def _run_one_epoch(self, epoch, loader, is_train=True):
        epoch_report = defaultdict(float)
        if is_train:
            progress_bar = tqdm(
                enumerate(loader), total=len(loader),
                desc="Epoch {}".format(epoch), ncols=0)
        else:
            progress_bar = enumerate(loader)
        with torch.set_grad_enabled(is_train):
            for i, data in progress_bar:
                self.callbacks.on_batch_begin(i)
                step_report = self._make_step(data, is_train)

                for key, value in step_report.items():
                    epoch_report[key] += value

                if is_train:
                    progress_bar.set_postfix(**{k: "{:.5f}".format(v.item() / (i + 1)) for k, v in epoch_report.items()})

                self.callbacks.on_batch_end(i, step_report=step_report, is_train=is_train)
        return {key: value.item() / len(loader) for key, value in epoch_report.items()}

    def _make_step(self, data, is_train):
        report = {}
        images, labels = self.batch_handler(data)

        if is_train:
            self.optimizer.zero_grad()

        predictions = self.model(images)
        loss = self.criterion(predictions, *labels)
        report['loss'] = loss.data

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return report

    def fit(self, train_loader, val_loader=None):
        try:
            self.callbacks.on_train_begin()
            for epoch in range(self.epochs):
                self.callbacks.on_epoch_begin(epoch)

                if self.scheduler is not None:
                    self.scheduler.step()

                self.model.train()
                self.metrics.train_metrics = self._run_one_epoch(epoch, train_loader)

                if val_loader is not None:
                    self.model.eval()
                    self.metrics.val_metrics = self._run_one_epoch(epoch, val_loader, is_train=False)
                    print(" | ".join("{}: {:.5f}".format(k, v) for k, v in self.metrics.val_metrics.items()))

                self.callbacks.on_epoch_end(epoch)
            self.callbacks.on_train_end()
        except KeyboardInterrupt:
            print('done.')
