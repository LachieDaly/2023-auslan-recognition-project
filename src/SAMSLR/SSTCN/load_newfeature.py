import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import scipy.io
import os
from torch.autograd import Variable
import random
 
class TorchDataset(Dataset):
    def __init__(self, is_train, feature_dir, is_aug=False, repeat=1):
        self.label_path = './Data/ELAR/train_val_labels.csv'
        self.is_train = is_train
        self.is_aug = is_aug
        # file name of training and testing samples
        self.feature_label_list = self.read_file()
        if self.is_train:
           random.shuffle(self.feature_label_list)
        self.feature_dir = feature_dir
        self.len = len(self.feature_label_list)
        self.repeat = repeat
        
    def __getitem__(self, i):
        index = i % self.len
        feature_name, label = self.feature_label_list[index]
        fea_path = os.path.join(self.feature_dir, feature_name)
        features = self.load_data(fea_path)
        label = np.array(label)
        return features, label

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.feature_label_list) * self.repeat
        return data_len

    def read_file(self):
        feature_label_list = []
        label_file = open(self.label_path, 'r', encoding='utf-8')
        for line in label_file.readlines():
            line = line.strip()
            line = line.split(',')

            if line[2] == "train" and self.is_train:
                name = line[0] + '.pt'
                label = int(line[1])
                feature_label_list.append((name, label))
            elif line[2] == "val" and not self.is_train:
                name = line[0] + '.pt'
                label = int(line[1])
                feature_label_list.append((name, label))

        return feature_label_list
 
    def load_data(self, path):
        data = torch.load(path, map_location='cpu')
        if self.is_aug:
            data = data.view(60,-1,24,24)
            judge = random.randint(0,12)
############## aug on frames ##########################################
            slist = range(0,60)
            if judge > 7.5 and judge < 11.5:
                rlength = 60 - random.randint(1,29)
                rindex = random.sample(range(0,60),rlength)
                extlist = random.sample(rindex,60-rlength)
                final_list = sorted([*rindex, *extlist])
                slist = np.array(final_list)

            if judge >2.5 and judge <3.5:
                rlength = 60 - random.randint(31,45)
                repeatnum =int (60/rlength)
                extension = 60 - rlength*repeatnum
                rindex = random.sample(range(0,60),rlength)
                extlist = random.sample(rindex,extension)
                rindex = list(np.repeat(np.array(rindex),repeatnum))
                final_list = sorted([*rindex, *extlist])
                slist = np.array(final_list)
            slist = list(slist)
##########################################################################
            if self.is_train:
                data = data[slist,:,:,:]
                data = data.view(-1,24,24)
            else:
                data = data.view(-1,24,24)
        else:
            data = data.view(-1,24,24)
        return data