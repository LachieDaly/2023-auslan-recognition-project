import cv2
import numpy as np
import os
from wholepose.utils import plot_pose

def crop(image, center, radius, frame_length, frame_height):
    """
    Crops an image 

    :param image: image to be cropped
    :param center: the new center of the image
    :param radius: the radius from the center to remain within the crop
    :return: cropped image
    """
    scale = 1.3
    radius_crop = (radius * scale).astype(np.int32)
    center_crop = (center).astype(np.int32)

    rect = (max(0,(center_crop-radius_crop)[0]), max(0,(center_crop-radius_crop)[1]), 
                 min(frame_length,(center_crop+radius_crop)[0]), min(frame_height,(center_crop+radius_crop)[1]))

    image = image[rect[1]:rect[3],rect[0]:rect[2],:]

    if image.shape[0] < image.shape[1]:
        top = abs(image.shape[0] - image.shape[1]) // 2
        bottom = abs(image.shape[0] - image.shape[1]) - top
        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT,value=(0,0,0))
    elif image.shape[0] > image.shape[1]:
        left = abs(image.shape[0] - image.shape[1]) // 2
        right = abs(image.shape[0] - image.shape[1]) - left
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT,value=(0,0,0))
    return image

selected_joints = np.concatenate(([0,1,2,3,4,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0) 

folder = './Data/ELAR/avi/train' # 'train', 'test'
npy_folder = './Data/ELAR/npy3/train' # 'train_npy/npy3', 'test_npy/npy3'
out_folder = './Data/ELAR/sam_frames_crop/train' # 'train_frames' 'test_frames'



for root, dirs, files in os.walk(folder, topdown=False):
    for name in files:
        if '.avi' in name:
            video_path = os.path.join(root, name)
            print(video_path)
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            frame_height = frame.shape[0]
            frame_length = frame.shape[1]
            cap.release()
            cap = cv2.VideoCapture(video_path)
            npy = np.load(os.path.join(npy_folder, name[:-4] + '.npy')).astype(np.float32)
            npy = npy[:, selected_joints, :2]
            npy[:, :, 0] = (frame_length/512) * (512 - npy[:, :, 0])
            npy[:, :, 1] = (frame_height/512) * npy[:, :, 1]
            xy_max = npy.max(axis=1, keepdims=False).max(axis=0, keepdims=False)
            xy_min = npy.min(axis=1, keepdims=False).min(axis=0, keepdims=False)
            assert xy_max.shape == (2,)
            xy_center = (xy_max + xy_min) / 2 - 20
            xy_radius = (xy_max - xy_center).max(axis=0)
            # window_name = "plots"
            index = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    image = crop(frame, xy_center, xy_radius, frame_length, frame_height)
                else:
                    break
                index = index + 1
                image = cv2.resize(image, (256,256))
                if not os.path.exists(os.path.join(out_folder, name[:-4])):
                    os.makedirs(os.path.join(out_folder, name[:-4]))
                cv2.imwrite(os.path.join(out_folder, name[:-4], '{:04d}.jpg'.format(index)), image)
                print(os.path.join(out_folder, name[:-4], '{:04d}.jpg'.format(index)))

cv2.destroyAllWindows()