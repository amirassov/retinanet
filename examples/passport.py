import cv2
import json
import torch
import argparse
import pandas as pd
from copy import copy
from torch.utils.data import DataLoader

from fpnssd.albumentations import (
    ToGray, Resize, ToTensor, Normalize, BBoxesToCoords, ChannelShuffle,
    CLAHE, Blur, HueSaturationValue, ShiftScaleRotate, CoordsToBBoxes,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAASharpen, IAAEmboss,
    RandomContrast, RandomBrightness, OneOf, Compose, ToAbsoluteCoords)
from fpnssd.train import Runner
from fpnssd.config import SSDConfig
from fpnssd.dataset import SSDDataset
from fpnssd.bboxer import BBoxTransform
from fpnssd.utils import set_global_seeds
from fpnssd.train.callbacks import ModelSaver

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--samples', type=str, default=None)
    parser.add_argument('--folds', type=str, default=None)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default=None)
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
        ToAbsoluteCoords(),
        BBoxesToCoords(),
        ShiftScaleRotate(
            x_shift_limit=0.0625,
            y_shift_limit=0.0625,
            scale_limit=(-0.3, 0.2),
            rotate_limit=5,
            border_mode=cv2.BORDER_CONSTANT),
        CoordsToBBoxes(),
        Resize(*image_size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor(),
    ], p=1.0)
    test_transform = Compose([
        ToAbsoluteCoords(),
        Resize(*image_size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
            if folds[_id]['fold'] == args.fold:
                val_samples.append(sample)
            else:
                train_samples.append(sample)
    return train_samples, val_samples


def batch_handler(data, device):
    images = data['image'].to(device)
    labels = [data['bboxes'].to(device), data['labels'].to(device)]
    return images, labels


def main():
    args = parse_args()
    set_global_seeds(args.seed)
    config = SSDConfig(args.config)

    train_samples, val_samples = split_samples(args)
    train_transform, val_transform = get_augmentation(copy(config.model.bboxer).cpu(), config.image_size)

    train_dataset = SSDDataset(
        class2label=config.class2label,
        samples=train_samples,
        transform=train_transform,
        data_dir=args.data_dir)

    val_dataset = SSDDataset(
        class2label=config.class2label,
        samples=val_samples,
        transform=val_transform,
        data_dir=args.data_dir)

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
        num_workers=config.data_params['num_workers'],
        pin_memory=torch.cuda.is_available())

    trainer = Runner(
        model=config.model,
        loss=config.loss,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        name=config.train_params['name'],
        epochs=config.train_params['epochs'],
        model_dir=config.train_params['model_dir'],
        callbacks=[ModelSaver(1, "best.pt", best_only=True)],
        batch_handler=batch_handler,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epoch_size=len(train_loader))
    trainer.fit(train_loader, val_loader)


if __name__ == '__main__':
    main()
