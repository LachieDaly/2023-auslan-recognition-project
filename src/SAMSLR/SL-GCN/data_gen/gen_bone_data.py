import numpy as np
from numpy.lib.format import open_memmap

paris = {
    'sign/27': ((5, 6), (5, 7),
                (6, 8), (8, 10), (7, 9), (9, 11), 
                (12,13),(12,14),(12,16),(12,18),(12,20),
                (14,15),(16,17),(18,19),(20,21),
                (22,23),(22,24),(22,26),(22,28),(22,30),
                (24,25),(26,27),(28,29),(30,31),
                (10,12),(11,22)
    ),
}

sets = {
    'train', 'val'
}

datasets = {
    'sign/27'
}

from tqdm import tqdm

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data = np.load('./Data/ELAR/{}/{}_data_joint.npy'.format(dataset, set))
        # N Samples
        # C  positions, y positions, confidences
        # T  frame number
        # V  joint number
        # M  joint dimension
        N, C, T, V, M = data.shape
        # print(N, C, T, V, M)
        # print(data[0][0][0][0])
        # print(data[0][1][0][0])
        # print(data[0][2][0][0])
        """Changing this will change it in our memory file as well"""
        fp_sp = open_memmap(
            './Data/ELAR/{}/{}_data_bone.npy'.format(dataset, set),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))

        fp_sp[:, :C, :, :, :] = data
        # Start to build the bone vectors
        for v1, v2 in tqdm(paris[dataset]):
            v1 -= 5
            v2 -= 5
            fp_sp[:, :, :, v2, :] = data[:, :, :, v2, :] - data[:, :, :, v1, :]