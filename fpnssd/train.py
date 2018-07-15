import os
from collections import defaultdict
from copy import deepcopy

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm


class PytorchTrain:
    def __init__(self, model, epochs, loss, name, model_dir, log_dir, metrics, optimizer, scheduler):
        self.model = torch.nn.DataParallel(model).cuda()
        self.epochs = epochs
        self.name = name
        self.log_dir = os.path.join(log_dir, self.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model_dir = os.path.join(model_dir, self.name)
        os.makedirs(self.model_dir, exist_ok=True)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = loss.cuda()
        self.writer = SummaryWriter(self.log_dir)
        self.metrics = metrics

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
        multi_bboxes = data['bboxes'].cuda()
        multi_labels = data['labels'].cuda()

        if is_train:
            self.optimizer.zero_grad()

        multi_bbox_predictions, multi_label_predictions = self.model(images)
        name2loss = self.criterion(multi_bbox_predictions, multi_bboxes, multi_label_predictions, multi_labels)

        for name, loss in name2loss.items():
            report[name] = loss.data

        for name, func in self.metrics:
            report[name] = func(multi_bbox_predictions, multi_bboxes, multi_label_predictions, multi_labels).data

        if is_train:
            name2loss['main_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return report

    def fit(self, train_loader, val_loader):
        best_epoch = -1
        best_loss = float('inf')
        try:
            for epoch in range(self.epochs):
                self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    print(param_group['lr'])

                self.model.train()
                train_metrics = self._run_one_epoch(epoch, train_loader)

                self.model.eval()
                val_metrics = self._run_one_epoch(epoch, val_loader, is_train=False)

                print(" | ".join("{}: {:.5f}".format(key, float(value)) for key, value in val_metrics.items()))

                for key, value in train_metrics.items():
                    self.writer.add_scalar('train/{}'.format(key), float(value), global_step=epoch)

                for key, value in val_metrics.items():
                    self.writer.add_scalar('val/{}'.format(key), float(value), global_step=epoch)

                loss = float(val_metrics['main_loss'])

                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    torch.save(deepcopy(self.model), os.path.join(self.model_dir, 'best.pt'))

        except KeyboardInterrupt:
            print('done.')

        self.writer.close()
        print('Finished: best losses {:.5f} on epoch {}'.format(best_loss, best_epoch))
