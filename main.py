import os
import cv2
import sys
import json
import torch
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
from fpnssd.config import read_config, OPTIMIZERS, LOSSES
from fpnssd.train import PytorchTrain
from fpnssd.dataset import SSDDataset
from fpnssd.utils import set_global_seeds
from fpnssd.models import SSD

cv2.ocl.setUseOpenCL(False)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--samples', type=str, default=None)
    parser.add_argument('--folds', type=str, default=None)
    parser.add_argument('--val_fold', type=int, default=0)
    return parser.parse_args()


def get_augmentation(mode, bbox_encoder):
    if mode == 'hard':
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
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT),
            CoordsToBBoxes(),
            Resize(min_dim=512, max_dim=512),
            ToTensor(),
            bbox_encoder
        ], p=1.0)
        test_transform = Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToAbsoluteCoords(),
            Resize(min_dim=512, max_dim=512),
            ToTensor(),
            bbox_encoder
        ], p=1.0)
        return train_transform, test_transform


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
    config = read_config(args.config)
    model = SSD(
        class2label=config['class2label'],
        bbox_kwargs=config['bbox_kwargs'],
        feature_extracter_kwargs=config['feature_extracter_kwargs'])

    train_samples, val_samples = split_samples(args)
    train_transform, val_transform = get_augmentation(config['augmentation'], model.bboxer.encoder)

    train_dataset = SSDDataset(class2label=config['class2label'], samples=train_samples, transform=train_transform)
    val_dataset = SSDDataset(class2label=config['class2label'], samples=val_samples, transform=val_transform)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=config['batch_size'], shuffle=True,
        drop_last=True, num_workers=config['num_workers'], pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(
        dataset=val_dataset, batch_size=config['batch_size'], shuffle=False,
        drop_last=False, num_workers=config['num_workers'], pin_memory=torch.cuda.is_available())

    loss = LOSSES[config['train']['loss']](num_classes=len(config['classes']) + 1)
    optimizer = OPTIMIZERS[config['train']['optimizer']]
    trainer = PytorchTrain(
        model=model,
        loss=loss,
        optimizer=optimizer,
        name=config['name'],
        epochs=config['train']['epochs'],
        model_dir=config['train']['model_dir'],
        log_dir=config['train']['log_dir'],
        metrics=config['train']['metrics'],
        lr=config['train']['lr'])

    trainer.fit(train_loader, val_loader)


if __name__ == '__main__':
    main()
