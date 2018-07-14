import os
import cv2
import sys
import json
import torch
from copy import copy
import argparse
import pandas as pd
from torch.utils.data import DataLoader

from fpnssd.albumentations import (
    ToGray, Resize, ToTensor, Normalize, BBoxesToCoords, ChannelShuffle,
    CLAHE, Blur, HueSaturationValue, ShiftScaleRotate, CoordsToBBoxes,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAASharpen, IAAEmboss,
    RandomContrast, RandomBrightness, OneOf, Compose, ToAbsoluteCoords
)

sys.path.append("../fpnssd")
from fpnssd.config import SSDConfig
from fpnssd.train import PytorchTrain
from fpnssd.dataset import SSDDataset
from fpnssd.bboxer import BBoxTransform
from fpnssd.utils import set_global_seeds

cv2.ocl.setUseOpenCL(False)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--samples', type=str, default=None)
    parser.add_argument('--folds', type=str, default=None)
    parser.add_argument('--val_fold', type=int, default=0)
    return parser.parse_args()


def get_augmentation(bboxer, image_size):
    train_transform = Compose([
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.5),
        OneOf([
            MotionBlur(blur_limit=3, p=0.2),
            MedianBlur(blur_limit=3, p=1.0),
            Blur(blur_limit=3, p=0.1),
        ], p=0.5),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
        ], p=0.5),
        RandomContrast(),
        RandomBrightness(),
        HueSaturationValue(p=0.3),
        OneOf([
            ChannelShuffle(),
            ToGray()
        ], p=0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToAbsoluteCoords(),
        BBoxesToCoords(),
        ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=(-0.3, 0.2),
            rotate_limit=5,
            border_mode=cv2.BORDER_CONSTANT),
        CoordsToBBoxes(),
        Resize(*image_size, mode='square'),
        ToTensor(),
    ], p=1.0)
    test_transform = Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToAbsoluteCoords(),
        Resize(*image_size, mode='square'),
        ToTensor(),
    ], p=1.0)
    return BBoxTransform(train_transform, bboxer), BBoxTransform(test_transform, bboxer)


def split_samples(args):
    with open(args.samples) as stream:
        samples = json.load(stream)

    folds = pd.read_csv(args.folds, index_col='ids').T.to_dict()
    train_samples = []
    val_samples = []
    for sample in samples:
        _id = sample['id']
        if _id in folds:
            if folds[_id]['fold'] == args.val_fold:
                val_samples.append(sample)
            else:
                train_samples.append(sample)
    return train_samples, val_samples


def main():
    args = parse_args()
    set_global_seeds(args.seed)
    config = SSDConfig(args.config)

    train_samples, val_samples = split_samples(args)
    train_transform, val_transform = get_augmentation(copy(config.model.bboxer).cpu(), config.image_size)

    train_dataset = SSDDataset(
        class2label=config.class2label,
        samples=train_samples,
        transform=train_transform)

    val_dataset = SSDDataset(
        class2label=config.class2label,
        samples=val_samples,
        transform=val_transform)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.data_params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=config.data_params['num_workers'],
        pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.data_params['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=config.data_params['batch_size'],
        pin_memory=torch.cuda.is_available())

    trainer = PytorchTrain(
        model=config.model,
        loss=config.loss,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        name=config.train_params['name'],
        epochs=config.train_params['epochs'],
        model_dir=config.train_params['model_dir'],
        log_dir=config.train_params['log_dir'],
        metrics=config.train_params['metrics'])

    trainer.fit(train_loader, val_loader)


if __name__ == '__main__':
    main()
