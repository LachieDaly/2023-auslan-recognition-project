import json
import math
import os

import numpy as np
import torch
import torchvision
import cv2
import time

# from .transforms import Compose, Scale, MultiScaleCrop, ToFloatTensor, PermuteImage, Normalize, scales, NORM_STD_IMGNET, \
#     NORM_MEAN_IMGNET, CenterCrop, IMAGE_SIZE, DeleteFlowKeypoints, ColorJitter, RandomHorizontalFlip

from pathlib import Path

_DATA_DIR_LOCAL = './Data/Demo/mp4'# it has got to change

SHOULDER_DIST_EPSILON = 1.2
WRIST_DELTA = 0.15

# transform = Compose(Scale(IMAGE_SIZE * 8 // 7), CenterCrop(IMAGE_SIZE), ToFloatTensor(), PermuteImage(), Normalize(NORM_MEAN_IMGNET, NORM_STD_IMGNET))

# replace sample['path'] with something else

sample_path = ".\\Data\\Demo\\mp4\\train\\DemoVideo.mp4"
frames, _, _ = torchvision.io.read_video(sample_path, pts_unit='sec')
for frame_index in range(0, 127, 2):
    kp_path = os.path.join(sample_path.replace('mp4', 'kp'), '{}_{:012d}_keypoints.json'.format(
            sample_path.split('\\')[-1].replace('.mp4', ''), frame_index))

    with open(kp_path, 'r') as keypoints_file:
        value = json.loads(keypoints_file.read())
        keypoints = np.array(value['people'][0]['pose_keypoints_2d'])
        # Extract keypoints exclusing confidence values
        x = keypoints[0::3]
        y = keypoints[1::3]
        keypoints = np.stack((x, y), axis=0)

    poseflow = None
    frame_index_poseflow = frame_index
    if frame_index_poseflow > 0:
        full_path = os.path.join(sample_path.replace('mp4', 'kpflow2'),
                                    'flow_{:05d}.npy'.format(frame_index_poseflow))
        while not os.path.isfile(full_path):  # WORKAROUND FOR MISSING FILES!!!
            frame_index_poseflow -= 1
            full_path = os.path.join(sample_path.replace('mp4', 'kpflow2'),
                                        'flow_{:05d}.npy'.format(frame_index_poseflow))

        value = np.load(full_path)
        poseflow = value
        # Normalize the angle between -1 and 1 from -pi and pi
        poseflow[:, 0] /= math.pi
        # Magnitude is already normalized from the pre-processing done before calculating the flow
    else:
        poseflow = np.zeros((135, 2))

    frame = frames[frame_index]
    cv2.imshow("Frame", frame.numpy())

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
        cv2.imshow("Left Hand Crop", left_hand_crop.numpy())
        # missing_wrists_left.append(len(clip) + 1)
    else:
        left_hand_crop = frame[left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :]
        cv2.imshow("Left Hand Crop", left_hand_crop.numpy())
    # left_hand_crop = transform(left_hand_crop.numpy())

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
        cv2.imshow("Right Hand Crop", right_hand_crop.numpy())
        # missing_wrists_right.append(len(clip) + 1)
    else:
        right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :]
        cv2.imshow("Right Hand Crop", right_hand_crop.numpy())
    # right_hand_crop = transform(right_hand_crop.numpy())

    # crops = torch.stack((left_hand_crop, right_hand_crop), dim=0)
    if cv2.waitKey(1) & 0xFF == 27:
        break;

    # clip.append(crops)
    time.sleep(1)

cv2.destroyAllWindows()


def draw_poseflow(image, poseflow, pose):
    # Then see how it goes