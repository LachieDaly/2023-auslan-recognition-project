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

    def val_dataloader(self, return_path=True):
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
                          pin_memory=True                          
                          )

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
        missing_wrists_left, missing_wrists_right = [], []
        for frame_index in sample['frames']:
            kp_path = os.path.join(sample['path'].replace('avi', 'kp'), '{}_{:012d}_keypoints.json'.format(
                    sample['path'].split('\\')[-1].replace('.avi', ''), frame_index))

            with open(kp_path, 'r') as keypoints_file:
                value = json.loads(keypoints_file.read())
                keypoints = np.array(value['people'][0]['pose_keypoints_2d'])
                x = keypoints[0::3]
                y = keypoints[1::3]
                keypoints = np.stack((x, y), axis=0)

            frame = frames[frame_index]

            left_wrist_index = 9
            left_elbow_index = 7
            right_wrist_index = 10
            right_elbow_index = 8

            # Crop out both wrists and apply transform
            left_wrist = keypoints[0:2, left_wrist_index]
            left_elbow = keypoints[0:2, left_elbow_index]
            left_hand_center = left_wrist + WRIST_DELTA * (left_wrist - left_elbow)
            left_hand_center_x = left_hand_center[0]
            left_hand_center_y = left_hand_center[1]
            shoulder_dist = np.linalg.norm(keypoints[0:2, 5] - keypoints[0:2, 6]) * SHOULDER_DIST_EPSILON
            left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
            left_hand_xmax = min(frame.size(1), int(left_hand_center_x + shoulder_dist // 2))
            left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
            left_hand_ymax = min(frame.size(0), int(left_hand_center_y + shoulder_dist // 2))

            if not np.any(left_wrist) or not np.any(
                    left_elbow) or left_hand_ymax - left_hand_ymin <= 0 or left_hand_xmax - left_hand_xmin <= 0:
                # Wrist or elbow not found -> use entire frame then
                left_hand_crop = frame
                missing_wrists_left.append(len(clip) + 1)
            else:
                left_hand_crop = frame[left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :]
            left_hand_crop = self.transform(left_hand_crop.numpy())

            right_wrist = keypoints[0:2, right_wrist_index]
            right_elbow = keypoints[0:2, right_elbow_index]
            right_hand_center = right_wrist + WRIST_DELTA * (right_wrist - right_elbow)
            right_hand_center_x = right_hand_center[0]
            right_hand_center_y = right_hand_center[1]
            right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
            right_hand_xmax = min(frame.size(1), int(right_hand_center_x + shoulder_dist // 2))
            right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
            right_hand_ymax = min(frame.size(0), int(right_hand_center_y + shoulder_dist // 2))

            if not np.any(right_wrist) or not np.any(
                    right_elbow) or right_hand_ymax - right_hand_ymin <= 0 or right_hand_xmax - right_hand_xmin <= 0:
                # Wrist or elbow not found -> use entire frame then
                right_hand_crop = frame
                missing_wrists_right.append(len(clip) + 1)
            else:
                right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :]
            right_hand_crop = self.transform(right_hand_crop.numpy())

            crops = torch.stack((left_hand_crop, right_hand_crop), dim=0)

            clip.append(crops)

        # Try to impute hand crops from frames where the elbow and wrist weren't missing as close as possible temporally
        for clip_index in range(len(clip)):
            if clip_index in missing_wrists_left:
                # Find temporally closest not missing frame for left wrist
                replacement_index = -1
                distance = np.inf
                for ci in range(len(clip)):
                    if ci not in missing_wrists_left:
                        dist = abs(ci - clip_index)
                        if dist < distance:
                            distance = dist
                            replacement_index = ci
                if replacement_index != -1:
                    clip[clip_index][0] = clip[replacement_index][0]
            # Same for right crop
            if clip_index in missing_wrists_right:
                # Find temporally closest not missing frame for right wrist
                replacement_index = -1
                distance = np.inf
                for ci in range(len(clip)):
                    if ci not in missing_wrists_right:
                        dist = abs(ci - clip_index)
                        if dist < distance:
                            distance = dist
                            replacement_index = ci
                if replacement_index != -1:
                    clip[clip_index][1] = clip[replacement_index][1]

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
