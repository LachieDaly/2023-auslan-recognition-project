import os
import numpy as np
from numpy.lib.format import open_memmap

sets = {
    'train', 'val'
}

datasets = {
    'sign/27'
}

parts = {
    'joint', 'bone'
}
from tqdm import tqdm

for dataset in datasets: # only looking to do this for 27 keypoints
    for set in sets: # both training and validation sets
        for part in parts: # both joint and bone motion to calculate
            print(dataset, set, part)
            data = np.load('./Data/ELAR/{}/{}_data_{}.npy'.format(dataset, set, part))
            N, C, T, V, M = data.shape
            print(data.shape)
            fp_sp = open_memmap(
                './Data/ELAR/{}/{}_data_{}_motion.npy'.format(dataset, set, part),
                dtype='float32',
                mode='w+',
                shape=(N, C, T, V, M))
            for t in tqdm(range(T - 1)):
                # Start to build the motions vectors between frames
                fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
            fp_sp[:, :, T - 1, :, :] = 0