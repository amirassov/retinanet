import os
import cv2
import json
import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader

from fpnssd.albumentations import (
    ToGray, Resize, ToTensor, Normalize,
    CLAHE, Blur, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAASharpen, IAAEmboss,
    RandomContrast, RandomBrightness, OneOf, Compose, ToAbsoluteCoords
)
from fpnssd.utils import read_config
from fpnssd.train import PytorchTrain
from fpnssd.dataset import SSDDataset
from fpnssd.utils import set_global_seeds
from fpnssd.models import BBoxer, FPNSSD

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


def get_augmentation(mode):
    if mode == 'hard':
        train_transform = Compose([
            Resize(shape=(128, 128), p=1.0),
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
                RandomContrast(),
                RandomBrightness(),
            ], p=0.5),
            HueSaturationValue(p=0.3),
            ToGray(),
            ToAbsoluteCoords(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
            ToTensor()
        ], p=1.0)
        test_transform = Compose([
            Resize(shape=(128, 128), p=1.0),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
            ToAbsoluteCoords(),
            ToTensor()
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
    box_coder = BBoxer(
        image_size=config['image_size'],
        anchor_areas=config['anchor_areas'],
        aspect_ratios=config['aspect_ratios'],
        scale_ratios=config['scale_ratios'],
        backbone_strides=config['backbone_strides']
    )

    train_samples, val_samples = split_samples(args)
    train_transform, val_transform = get_augmentation(config['augmentation'])

    train_dataset = SSDDataset(samples=train_samples, transform=train_transform, box_coder=box_coder)
    val_dataset = SSDDataset(samples=val_samples, transform=val_transform, box_coder=box_coder)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=config['batch_size'], shuffle=True,
        drop_last=True, num_workers=config['num_workers'], pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        dataset=val_dataset, batch_size=config['batch_size'], shuffle=False,
        drop_last=False, num_workers=config['num_workers'], pin_memory=torch.cuda.is_available()
    )

    model = FPNSSD(
        num_classes=config['num_classes'],
        num_anchors=box_coder.num_anchors,
        encoder=config['encoder'],
        encoder_args=config['encoder_args']
    )

    trainer = PytorchTrain(
        model=model,
        epochs=config['epochs'],
        loss=config['loss'],
        model_dir=config['model_dir'],
        log_dir=config['log_dir'],
        metrics=config['metrics'],
        loss_args=config['loss_args'],
        optimizer=config['optimizer'],
        optimizer_args=config['optimizer_args']
    )

    trainer.fit(train_loader, val_loader)


if __name__ == '__main__':
    main()
