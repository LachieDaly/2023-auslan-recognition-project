import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np

"""
Implementation of Sign Language Dataset
"""
class Sign_Isolated(Dataset):
    """
    A representation of our dataset for training and validation
    """
    def __init__(self, data_path, label_path, frames=16, num_classes=29, train=True, transform=None, test_clips=5):
        super(Sign_Isolated, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.train = train
        self.transform = transform
        self.frames = frames
        self.num_classes = num_classes
        self.test_clips = test_clips
        self.sample_names = []
        self.labels = []
        self.data_folder = []
        if train:
            job = "train"
        else:
            job = "val"
        label_file = open(label_path, 'r', encoding='utf-8')
        for line in label_file.readlines():
            line = line.strip()
            line = line.split(',')

            if line[2] == job:
                self.sample_names.append(line[0])
                self.data_folder.append(os.path.join(data_path, line[0]))
                self.labels.append(int(line[1]))

    def frame_indices_transform(self, video_length, sample_duration):
        """
        Gets a random segment equal to our sample duration if it can
        otherwise gets the entire video and if the video is too short
        the final frame is repeated
        """
        frame_start = (video_length - sample_duration) // (2)
        frame_end = frame_start + sample_duration
        if frame_start < 0:
            frame_start = 0

        if frame_end > video_length:
            frame_end = video_length
        
        frame_indices = np.arange(frame_start, frame_end, 1)
        while len(frame_indices) < sample_duration:
            frame_indices = np.append(frame_indices, frame_indices[-1])
        # if video_length > sample_duration:
        #     random_start = random.randint(0, video_length - sample_duration)
        #     frame_indices = np.arange(random_start, random_start + sample_duration) + 1
        # else:
        #     frame_indices = np.arange(video_length)
        #     while frame_indices.shape[0] < sample_duration:
        #         # seem to repeat 
        #         # frame_indices = np.concatenate((frame_indices, np.arange(video_length)), axis=0)
        #         frame_indices = np.concatenate((frame_indices, np.array([frame_indices[-1]])), axis=0)
        #     frame_indices = frame_indices[:sample_duration] + 1
        assert frame_indices.shape[0] == sample_duration
        return frame_indices + 1

    def frame_indices_transform_test(self, video_length, sample_duration, clip_no=0):
        frame_start = (video_length - sample_duration) // (2)
        frame_end = frame_start + sample_duration
        if frame_start < 0:
            frame_start = 0

        if frame_end > video_length:
            frame_end = video_length
        
        frame_indices = np.arange(frame_start, frame_end, 1)
        while len(frame_indices) < sample_duration:
            frame_indices = np.append(frame_indices, frame_indices[-1])
        # if video_length > sample_duration:
        #     random_start = random.randint(0, video_length - sample_duration)
        #     frame_indices = np.arange(random_start, random_start + sample_duration) + 1
        # else:
        #     frame_indices = np.arange(video_length)
        #     while frame_indices.shape[0] < sample_duration:
        #         # seem to repeat 
        #         # frame_indices = np.concatenate((frame_indices, np.arange(video_length)), axis=0)
        #         frame_indices = np.concatenate((frame_indices, np.array([frame_indices[-1]])), axis=0)
        #     frame_indices = frame_indices[:sample_duration] + 1
        assert frame_indices.shape[0] == sample_duration
        return frame_indices + 1
        # if video_length > sample_duration:
        #     start = (video_length - sample_duration) // (self.test_clips - 1) * clip_no
        #     frame_indices = np.arange(start, start + sample_duration) + 1
        # elif video_length == sample_duration:
        #     frame_indices = np.arange(sample_duration) + 1
        # else:
        #     frame_indices = np.arange(video_length)
        #     while frame_indices.shape[0] < sample_duration:
        #         # frame_indices = np.concatenate((frame_indices, np.arange(video_length)), axis=0)
        #         frame_indices = np.concatenate((frame_indices, np.array([frame_indices[-1]])), axis=0)
        #     frame_indices = frame_indices[:sample_duration] + 1

        # return frame_indices

    def random_crop_paras(self, input_size, output_size):
        """
        Returns set of coord required to randomly crop images
        """
        diff = input_size - output_size
        i = random.randint(0, diff)
        j = random.randint(0, diff)
        return i, j, i+output_size, j+output_size

    def read_images(self, folder_path, clip_no=0):
        """
        Reads images from our image folder
        """
        # assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder: " + str(folder_path)
        images = []
        if self.train:
            index_list = self.frame_indices_transform(len(os.listdir(folder_path)), self.frames)
            flip_rand = random.random()
            angle = (random.random() - 0.5) * 10
            crop_box = self.random_crop_paras(256, 224)
        else:
            index_list = self.frame_indices_transform_test(len(os.listdir(folder_path)), self.frames, clip_no)
        
        # for i in range(self.frames):
        for i in index_list:
            imagePath = os.path.join(folder_path, '{:04d}.jpg'.format(i))
            image = Image.open(imagePath)
            if self.train:
                if flip_rand > 0.5:
                    image = ImageOps.mirror(image)
                image = transforms.functional.rotate(image, angle)
                image = image.crop(crop_box)
                assert image.size[0] == 224
            else:
                crop_box = (16, 16, 240, 240)
                image = image.crop(crop_box)
                # assert image.size[0] == 224
            if self.transform is not None:
                image = self.transform(image)

            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension for 3d cnn
        images = images.permute(1, 0, 2, 3)
        return images

    def __len__(self):
        """
        Returns the number of samples in the particular training/validation/test set
        """
        return len(self.data_folder)

    def __getitem__(self, idx):
        """
        Get Item for our DataLoader
        """
        selected_folder = self.data_folder[idx]
        if self.train:
            images = self.read_images(selected_folder)
        else:
            images = []
            for i in range(self.test_clips):
                images.append(self.read_images(selected_folder, i))
            images = torch.stack(images, dim=0)
            # M, T, C, H, W
        label = torch.LongTensor([self.labels[idx]])
        return {'data': images, 'label': label}