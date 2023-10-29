import json
import math
import os

import numpy as np
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

def draw_poseflow(img, pose, poseflow, scale=(1.0, 1.0)):
    # Then see how it goes
    """
    Given image and pose information, draws pose on image

    :param img: image to draw pose on
    :param result: pose information
    :param scale: scale for keypoints
    :return: image pose keypoints and lines
    """
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255)
    PINK = (203, 192, 255)
    unshown_pts = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    l_pair = [         #v3 web
        #coco pose
        
        (5, 7),# R shoulder - R elbow
        (7, 9),# R elbow - R wrist
        
        (6, 8), # M shoulder - L shoulder
        (8, 10),# L shoulder - L elbow
        (5, 6),# L elbow - L wrist
        
#====================================
#=========feet ===================

#========================================
#=========face===========================
        (23, 24),#contour
        (24, 25),
        (25, 26),
        (26, 27),
        (27, 28),
        (28, 29),
        (29, 30),
        (30, 31),
        (31, 32),
        (32, 33),
        (33, 34),
        (34, 35),
        (35, 36),
        (36, 37),
        (37, 38),
        (38, 39),
        
        (40, 41),#R eye brows
        (41, 42),
        (42, 43),
        (43, 44),
        (45, 46),#L eye brows
        (46, 47),
        (47, 48),
        (48, 49),
        
        (50, 51), #nose
        (51, 52),
        (52, 53),
        (54, 55),
        (55, 56),
        (56, 57),
        (57, 58),
        
        (59, 60),# R eye
        (60, 61),
        (61, 62),
        (62, 63),
        (63, 64),
        (64, 59),
        (65, 66),#L eye
        (66, 67),
        (67, 68),
        (68, 69),
        (69, 70),
        (70, 65),
        
        (71, 72),#out mouth
        (72, 73),
        (73, 74),
        (74, 75),
        (75, 76),
        (76, 77),
        (77, 78),
        (78, 79),
        (79, 80),
        (80, 81),
        (81, 82),
        (82, 71),
        
        (83, 84),#in mouth
        (84, 85),
        (85, 86),
        (86, 87),
        (87, 88),
        (88, 89),
        (89, 90),
        (90, 83),
#==================================================
#=============hands================================
        (9, 91),#L wrist - L hand
        (91, 92),# L thumb
        (92, 93),
        (93, 94),
        (94, 95),
        (91, 96),#L index finger
        (96, 97),
        (97, 98),
        (98, 99),
        (91, 100), #L mid finger
        (100, 101),
        (101, 102),
        (102, 103),
        (91, 104), # L ring finger
        (104, 105),
        (105, 106),
        (106, 107),
        (91, 108), #L little finger
        (108, 109),
        (109, 110),
        (110, 111),
        
        (10, 112), # R wrist - R hand
        (112, 113),#R thumb
        (113, 114),
        (114, 115),
        (115, 116),
        (112, 117),#R index finger
        (117, 118),
        (118, 119),
        (119, 120),
        (112, 121),#R mid finger
        (121, 122),
        (122, 123),
        (123, 124),
        (112, 125),#R ring finger
        (125, 126),
        (126, 127),
        (127, 128),
        (112, 129), #R little finger
        (129, 130),
        (130, 131),
        (131, 132),
    ]
    p_color = [GREEN, BLUE, BLUE, BLUE, BLUE, YELLOW, ORANGE, YELLOW, ORANGE,
                YELLOW, ORANGE, PINK, RED, PINK, RED, PINK, RED]
    p_color = [GREEN, BLUE, BLUE, BLUE, BLUE, YELLOW, ORANGE, YELLOW, ORANGE,
                YELLOW, ORANGE, PINK, RED, PINK, RED, PINK]

    part_line = {}
    # kp_preds = result[:, :2]
    # kp_scores = result[:, 2]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #print(kp_scores.size())
    # Draw keypoints
    print(poseflow.shape)
    for n in range(pose.shape[1]):
        # if kp_scores[n] <= 0.1 or n in unshown_pts:
        #     continue
        cor_x, cor_y = int(round(pose[0, n] * scale[0])), int(round(pose[1, n] * scale[1]))
        print(cor_x, cor_y)
        end_x, end_y = int(round(cor_x + 200*poseflow[n, 1] * math.cos(poseflow[n, 0]))), int(round(cor_y + 200*poseflow[n, 1] * math.sin(poseflow[n, 0])))
        print(end_x, end_y)
        # part_line[n] = (cor_x, cor_y)
        cv2.circle(img, (cor_x, cor_y), 2, (0, 0, 255), 4)
        cv2.arrowedLine(img, (cor_x, cor_y), (end_x, end_y), (0, 255, 0), 2)
    # Draw limbs
    # for start_p, end_p in l_pair:
    #     if start_p in part_line and end_p in part_line:
    #         start_p = part_line[start_p]
    #         end_p = part_line[end_p]
    #         cv2.line(img, start_p, end_p, (0, 255, 0), 2)
    return img

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
        # poseflow[:, 0] /= math.pi
        # Magnitude is already normalized from the pre-processing done before calculating the flow
    else:
        poseflow = np.zeros((135, 2))

    frame = frames[frame_index]
    cv2.imshow("Frame", draw_poseflow(frame.numpy(), keypoints, poseflow))

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
        # cv2.imshow("Left Hand Crop", left_hand_crop.numpy())
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
        # cv2.imshow("Right Hand Crop", right_hand_crop.numpy())
        # missing_wrists_right.append(len(clip) + 1)
    else:
        right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :]
        cv2.imshow("Right Hand Crop", right_hand_crop.numpy())
    # right_hand_crop = transform(right_hand_crop.numpy())

    # crops = torch.stack((left_hand_crop, right_hand_crop), dim=0)
    if cv2.waitKey(1) & 0xFF == 27:
        break;

    # clip.append(crops)
    time.sleep(.2)

cv2.destroyAllWindows()
