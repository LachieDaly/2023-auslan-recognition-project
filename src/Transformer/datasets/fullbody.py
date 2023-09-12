import json
import math
import os
from argparse import ArgumentParser

import numpy as np
import lightning.pytorch as pl # fix this stuff
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from .common import collect_samples
from .transforms import Compose, Scale, MultiScaleCrop, ToFloatTensor, PermuteImage, Normalize, scales, NORM_STD_IMGNET, \
    NORM_MEAN_IMGNET, CenterCrop, IMAGE_SIZE, DeleteFlowKeypoints, ColorJitter, RandomHorizontalFlip

from pathlib import Path

_DATA_DIR_LOCAL = './Data/ELAR/avi'# it has got to change

SHOULDER_DIST_EPSILON = 1.2
WRIST_DELTA = 0.15

def get_datamodule_def():
    return ElarDataModule # we can definitely modify this morning


def get_datamodule(**kwargs):
    return ElarDataModule(**kwargs)


class ElarDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=_DATA_DIR_LOCAL, batch_size=16, num_workers=0, sequence_length=16,
                 temporal_stride=1, **kwargs):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.temporal_stride = temporal_stride

    def train_dataloader(self):
        transform = Compose(Scale(IMAGE_SIZE * 8 // 7), MultiScaleCrop((IMAGE_SIZE, IMAGE_SIZE), scales),
                            RandomHorizontalFlip(), ColorJitter(0.5, 0.5, 0.5),
                            ToFloatTensor(), PermuteImage(),
                            Normalize(NORM_MEAN_IMGNET, NORM_STD_IMGNET))
        
        self.train_set = ElarDataset(self.data_dir, 
                                         'train', 
                                         'train',
                                         self.data_dir / 'train_val_labels.csv',
                                         transform, 
                                         self.sequence_length, 
                                         self.temporal_stride)
        
        return DataLoader(self.train_set, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          pin_memory=True,
                          shuffle=True)

    def val_dataloader(self, return_path=False):
        transform = Compose(Scale(IMAGE_SIZE * 8 // 7), CenterCrop(IMAGE_SIZE), ToFloatTensor(),
                            PermuteImage(),
                            Normalize(NORM_MEAN_IMGNET, NORM_STD_IMGNET))
        
        self.val_set = ElarDataset(self.data_dir, 
                                   'train', 
                                   'val',
                                   self.data_dir / 'train_val_labels.csv',
                                   transform,
                                   self.sequence_length,
                                   self.temporal_stride,
                                   return_path=return_path)
        
        return DataLoader(self.val_set, 
                          batch_size=self.
                          batch_size, 
                          num_workers=self.num_workers, 
                          pin_memory=True)

    def test_dataloader(self):
        transform = Compose(Scale(IMAGE_SIZE * 8 // 7), CenterCrop(IMAGE_SIZE), ToFloatTensor(),
                            PermuteImage(),
                            Normalize(NORM_MEAN_IMGNET, NORM_STD_IMGNET))
        
        self.test_set = ElarDataset(self.data_dir, 
                                        'train', 
                                        'val', 
                                        self.data_dir / 'train_val_labels.csv', 
                                        transform, 
                                        self.sequence_length,
                                        self.temporal_stride)
        
        return DataLoader(self.test_set, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          pin_memory=True)

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--data_dir', type=str, default=_DATA_DIR_LOCAL)
        parser.add_argument('--sequence_length', type=int, default=16)
        parser.add_argument('--temporal_stride', type=int, default=2)
        return parser


class ElarDataset(Dataset):
    def __init__(self, root_path, job_path, job, label_file_path, transform, sequence_length,
                 temporal_stride, return_path=False):
        self.root_path = Path(root_path)
        self.job_path = job_path
        self.job = job
        self.label_file_path = label_file_path
        self.has_labels = self.label_file_path is not None
        self.return_path = return_path
        self.transform = transform
        self.sequence_length = sequence_length
        self.temporal_stride = temporal_stride

        self.samples = self._collect_samples()

    def __getitem__(self, item):
        self.transform.randomize_parameters()

        sample = self.samples[item]
        frames, _, _ = torchvision.io.read_video(sample['path'], pts_unit='sec')

        clip = []
        for frame_index in sample['frames']:
            frame = frames[frame_index]

            full_body_crop = self.transform(frame.numpy());

            clip.append(full_body_crop)

        clip = torch.stack(clip, dim=0)
        if not self.return_path:
            return clip, sample['label']
        else:
            # Return sample name instead of label so we know what prediction this is 
            return clip, sample['path'].split('\\')[-1][:-4]

    def __len__(self):
        return len(self.samples)

    def _collect_samples(self):
        return collect_samples(self.has_labels, self.root_path, self.job_path, self.sequence_length,
                               self.temporal_stride, self.job, self.label_file_path)
