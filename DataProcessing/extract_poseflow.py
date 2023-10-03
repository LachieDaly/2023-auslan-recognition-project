import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

def read_pose(kp_file):
    """
    Reads the json results of OpenPoseDemo and extracts the x, y coordinates of each keypoint
    Each index of the list is associated with its actual keypoint value
    Confidence values are removed
    """
    with open(kp_file) as kf:
        value = json.loads(kf.read())
        kps = value['people'][0]['pose_keypoints_2d']
        # Extract just x y values, leave behind confidence values
        x = kps[0::3]
        y = kps[1::3]
        return np.stack((x, y), axis=1)
    
def calc_pose_flow(prev, next):
    """
    Calculate the pose flow of each keypoint between frames
    """
    result = np.zeros_like(prev)
    for kpi in range(prev.shape[0]):
        if np.count_nonzero(prev[kpi]) == 0 or np.count_nonzero(next[kpi]) == 0:
            result[kpi, 0] = 0.0
            result[kpi, 1] = 0.0
            continue

        ang = math.atan2(next[kpi, 1] - prev[kpi, 1], next[kpi, 0] - prev[kpi, 0])
        mag = np.linalg.norm(next[kpi] - prev[kpi])

        result[kpi, 0] = ang
        result[kpi, 1] = mag

    return result

def impute_missing_keypoints(poses):
    """Replace missing keypoints (on the origin) by values from neighbouring frames."""
    # 1. Collect missing keypoints
    missing_keypoints = defaultdict(list) # frame index -> keypoint indices that are missing
    for i in range(poses.shape[0]):
        for kpi in range(poses.shape[1]):
            if np.count_nonzero(poses[i, kpi]) == 0: # Missing keypoint at (0, 0)
                missing_keypoints[i].append(kpi)
    
    # 2. Impute them
    for i in missing_keypoints.keys():
        missing = missing_keypoints[i]
        for kpi in missing:
            # possible replacements
            candidates = poses[:, kpi]
            min_dist = np.inf
            replacement = -1
            for f in range(candidates.shape[0]):
                if f != i and np.count_nonzero(candidates[f]) > 0:
                    distance = abs(f - i)
                    if distance < min_dist:
                        min_dist = distance
                        replacement = f
                    
            # Replace
            if replacement > -1:
                poses[i, kpi] = poses[replacement, kpi]
    # 3. We have imputed as many keypoints as possible with the closest non-missing temporal neighbours
    return poses

def normalise(poses):
    """
    Normalises each pose in the array to account for camera position. We normalise
    by dividing keypoints by a factor such that the lenght of the neck becomes 1.
    """
    for i in range(poses.shape[0]):
        upper_neck = poses[i, 17]
        head_top = poses[i, 18]
        neck_length = np.linalg.norm(upper_neck - head_top)
        poses[i] /= neck_length
        assert math.isclose(np.linalg.norm(upper_neck - head_top), 1)
    return poses

def main(args):
    """
    Convert all collected keypoints in a particular directory to their poseflow representation
    """
    input_dirs = sorted(glob.glob(os.path.join(args.input_dir, "*", "*.kp")))
    input_dir_index = 0
    total = len(input_dirs)
    for input_dir in input_dirs:
        input_dir_index += 1

        output_dir = input_dir.replace("kp", "kpflow2")
        os.makedirs(output_dir, exist_ok=True)

        kp_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

        # 1. Collect all keypoint files and pre-process them
        poses = []
        for i in range(len(kp_files)):
            poses.append(read_pose(kp_files[i]))
        poses = np.stack(poses)
        poses = impute_missing_keypoints(poses)
        poses = normalise(poses)

        # 2. Compute pose flow
        prev = poses[0]
        for i in range(1, poses.shape[0]):
            next = poses[i]
            flow = calc_pose_flow(prev, next)
            np.save(os.path.join(output_dir, "flow_{:05d}".format(i - 1)), flow)
            prev = next


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    args = parser.parse_args()
    main(args)