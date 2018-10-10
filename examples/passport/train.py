import cv2
import json
import torch
import argparse
import pandas as pd
from copy import copy
from torch.utils.data import DataLoader
import albumentations as alb
from albumentations.imgaug.transforms import IAAAffine
from albumentations.torch.transforms import ToTensor
from imgaug.parameters import Uniform

from fpnssd.train import Runner
from fpnssd.config import SSDConfig
from fpnssd.dataset import SSDDataset
from fpnssd.bboxer import BBoxTransform
from fpnssd.utils import set_global_seeds
from fpnssd.train.callbacks import ModelSaver, CheckpointSaver, TensorBoard

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
    bbox_conf = {
        'format': 'pascal_voc',
        'min_visibility': 0.7,
        'label_fields': ['labels']
    }
    augmentations = [
        alb.OneOf([
            alb.IAAAdditiveGaussianNoise(p=1),
            alb.GaussNoise(p=1),
        ]),
        alb.OneOf([
            alb.MotionBlur(blur_limit=3, p=1),
            alb.Blur(blur_limit=3, p=1),
        ]),
        alb.MedianBlur(),
        alb.OneOf([
            alb.CLAHE(clip_limit=2, p=1),
            alb.IAASharpen(p=1),
            alb.IAAEmboss(p=1),
        ]),
        alb.RandomContrast(),
        alb.RandomBrightness(),
        alb.HueSaturationValue(),
        alb.OneOf([
            alb.ChannelShuffle(p=1),
            alb.ToGray(p=1)
        ]),
        alb.JpegCompression(80, 99),
        alb.Flip(),
        alb.RandomRotate90(),
        IAAAffine(
            translate_percent=Uniform(-0.1, 0.1),
            rotate=Uniform(-45, 45),
            scale=Uniform(0.5, 2),
            mode='constant'
        )
    ]
    bbox_modification_augs = alb.Compose(augmentations, bbox_params=bbox_conf)

    post_trasforms = [
        alb.Resize(*image_size),
        alb.Normalize(),
        ToTensor()
    ]

    train_transform = alb.Compose([bbox_modification_augs] + post_trasforms)
    test_transform = alb.Compose(post_trasforms)
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
        callbacks=[ModelSaver(1, "best.pt", best_only=True),
                   CheckpointSaver(5, 'model_{epoch}.pth'),
                   TensorBoard(config.train_params['log_dir'])],
        batch_handler=batch_handler,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epoch_size=len(train_loader))
    trainer.fit(train_loader, val_loader)


if __name__ == '__main__':
    main()
