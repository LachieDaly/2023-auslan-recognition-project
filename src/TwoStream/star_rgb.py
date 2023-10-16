import os
import cv2
import glob
import numpy as np

def star_rbg_generation():
    # perform forward, backward, and concatenated fusion of the frames for ISA64 dataset where structure is [signer-->sign-->samples]
    home_source_folder = os.path.normpath('../dataset/elar/images/')
    signs = glob.glob(home_source_folder + '/*/*/*')
    target_path_star = os.path.normpath('../dataset/elar/star/')

    for sign in signs:
        # extract frames
        fpath = os.path.normpath(sign)
        filename = os.path.basename(fpath)
        file_path = os.path.dirname(fpath)
        new_path_star= os.path.normpath(file_path.replace(home_source_folder, target_path_star) + "/" + filename)
        sample_folder_full_path = fpath

        if not os.path.exists(os.path.join(new_path_star, filename)):
            os.makedirs(new_path_star, exist_ok=True)
            key_frames = glob.glob(sample_folder_full_path + '/*.png')
            # num_frames = len(list(key_frames))
            frame_array = []
            for key_frame in key_frames:
                frame_array.append(cv2.imread(key_frame))
            
            frame_number = len(frame_array)
            section_number = frame_number // 3
            extra_frames = frame_number % 3
            first_slice = section_number
            second_slice = 2 * section_number + extra_frames
            blue_range = frame_array[0:first_slice]
            green_range = frame_array[first_slice:second_slice]
            red_range = frame_array[second_slice:]
            blue_matrix = calculate_star(blue_range,1)
            green_matrix = calculate_star(green_range,2)
            red_matrix = calculate_star(red_range,3)
            star_representation = combine_matrices(blue_matrix, green_matrix, red_matrix)
            cv2.imwrite(os.path.join(new_path_star, f"{filename}.jpg"), star_representation)

def calculate_star(section, num):
    # print('calculate star')
    h = section[0].shape[0]
    w = section[0].shape[1]
    # print(h, w)
    result = np.zeros((h, w), np.uint8)
    for k in range(1, len(section)):
        current_frame_norm = np.linalg.norm(section[k], axis=2)
        previous_frame_norm = np.linalg.norm(section[k-1], axis=2)
        # Get distance
        euclidean = np.absolute(previous_frame_norm - current_frame_norm)
        # Get Angle
        multiplied = np.multiply(section[k-1], section[k])
        dot_product = np.sum(multiplied, axis=2)
        product_of_lengths = current_frame_norm*previous_frame_norm
        angle = 1 - ((dot_product)/(product_of_lengths))
        result = result + (1 - (angle / 2)) * euclidean
    return result


def combine_matrices(blue, green, red):
    image = np.stack((blue, green, red), axis=2)
    return image


star_rbg_generation()