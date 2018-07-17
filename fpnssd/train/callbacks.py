# Based on https://github.com/selimsef/dsb2018_topcoders/blob/master/albu/src/pytorch_utils/callbacks.py
import torch
from copy import deepcopy
import os
from tensorboardX import SummaryWriter


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        self.runner = None
        self.metrics_collection = None

    def set_trainer(self, runner):
        self.runner = runner
        self.metrics_collection = runner.metrics_collection

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass


class Callbacks(Callback):
    def __init__(self, callbacks):
        super().__init__()
        if isinstance(callbacks, Callbacks):
            callbacks = callbacks.callbacks
        self.callbacks = callbacks
        if callbacks is None:
            self.callbacks = []

    def set_trainer(self, runner):
        for callback in self.callbacks:
            callback.set_trainer(runner)

    def on_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch):
        for callback in self.callbacks:
            callback.on_batch_end(batch)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


class ModelSaver(Callback):
    def __init__(self, save_every, save_name, best_only=True):
        super().__init__()
        self.save_every = save_every
        self.save_name = save_name
        self.best_only = best_only

    def on_epoch_end(self, epoch):
        loss = float(self.metrics_collection.val_metrics['loss'])
        need_save = not self.best_only
        if epoch % self.save_every == 0:
            if loss < self.metrics_collection.best_loss:
                self.metrics_collection.best_loss = loss
                self.metrics_collection.best_epoch = epoch
                need_save = True

            if need_save:
                path = os.path.join(self.runner.model_dir, self.save_name).format(epoch=epoch, loss="{:.2}".format(loss))
                torch.save(obj=deepcopy(self.runner.model.module), f=path)


def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, path):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model_state_dict,
        'optimizer': optimizer_state_dict,
    }, path)


class CheckpointSaver(Callback):
    def __init__(self, save_every, save_name):
        super().__init__()
        self.save_every = save_every
        self.save_name = save_name

    def on_epoch_end(self, epoch):
        loss = float(self.metrics_collection.val_metrics['loss'])
        if epoch % self.save_every == 0:
            path = os.path.join(self.runner.model_dir, self.save_name).format(epoch=epoch, loss="{:.2}".format(loss))
            save_checkpoint(
                epoch=epoch,
                model_state_dict=self.runner.model.module.state_dict(),
                optimizer_state_dict=self.runner.optimizer.state_dict(),
                path=path)


class TensorBoard(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None

    def on_train_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def on_epoch_end(self, epoch):
        for k, v in self.metrics_collection.train_metrics.items():
            self.writer.add_scalar('train/{}'.format(k), float(v), global_step=epoch)

        for k, v in self.metrics_collection.val_metrics.items():
            self.writer.add_scalar('val/{}'.format(k), float(v), global_step=epoch)

        for idx, param_group in enumerate(self.runner.optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=epoch)

    def on_train_end(self):
        self.writer.close()
