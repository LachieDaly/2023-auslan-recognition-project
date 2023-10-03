import os
import cv2
import glob
from pathlib import Path
import numpy as np

def accumulative_video_motion():
    """
    Performs forwards, backwards, and concatenated fusion of the frames for The ELAR dataset
    placing each respective gesture in its own folder for training data
    """
    # perform forward, backward, and concatenated fusion of the frames for ISA64 dataset where structure is [signer-->sign-->samples]
    home_source_folder = os.path.normpath('../dataset/elar/images/')
    signs = glob.glob(home_source_folder + '/*/*/*')
    target_path_forward = os.path.normpath('../dataset/elar/fusion/forward/')
    target_path_backward = os.path.normpath('../dataset/elar/fusion/backward/')
    targetPath_both = os.path.normpath('../dataset/elar/fusion/both/')

    for sign in signs:
        # extract frames
        fpath = os.path.normpath(sign)
        filename = os.path.basename(fpath)
        file_path = os.path.dirname(fpath)
        new_path_forward = os.path.normpath(file_path.replace(home_source_folder, target_path_forward) + "/" + filename)
        new_path_backward = os.path.normpath(file_path.replace(home_source_folder, target_path_backward) + "/" + filename)
        new_path_both = os.path.normpath(file_path.replace(home_source_folder, targetPath_both) + "/" + filename)
        sample_folder_full_path = fpath

        if not os.path.exists(os.path.join(new_path_forward, filename)):
            os.makedirs(new_path_forward, exist_ok=True)
            os.makedirs(new_path_backward, exist_ok=True)
            os.makedirs(new_path_both, exist_ok=True)
            key_frames = glob.glob(sample_folder_full_path + '/*.png')
            num_frames = len(list(key_frames))

            if num_frames > 0:
                # Backward  
                f = num_frames - 1
                while f > 0:
                    image_frame_path = key_frames[f]
                    key_frame = cv2.imread(image_frame_path)
                    if f == num_frames - 1:
                        imgDiff = key_frame
                    else:
                        imgDiff = cv2.addWeighted(imgDiff, 0.5, key_frame, 0.5, 0)
                    f -= 1

                imgDiff_backward = imgDiff #saving image
                # Forward
                f = 0
                while f < num_frames:
                    image_frame_path = key_frames[f]
                    key_frame = cv2.imread(image_frame_path)
                    if f == 0:
                        imgDiff = key_frame
                    else:
                        imgDiff = cv2.addWeighted(imgDiff, 0.5, key_frame, 0.5, 0)
                    f += 1

                imgDiff_forward = imgDiff
                imgDiff_both = cv2.addWeighted(imgDiff_backward, 0.5, imgDiff_forward, 0.5, 0)

                cv2.imwrite(os.path.join(new_path_backward, f"{filename}.jpg"), imgDiff_backward)
                cv2.imwrite(os.path.join(new_path_forward, f"{filename}.jpg"), imgDiff_forward)
                cv2.imwrite(os.path.join(new_path_both, f"{filename}.jpg"), imgDiff_both)


accumulative_video_motion()