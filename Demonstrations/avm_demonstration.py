


import cv2
import numpy as np
from threading import Thread
import time

"""
This will be used to demonstrate how star works with data
"""

def rgb_star_representation(frames):
    """
    Calculate and return Star RGB representation of the provided frames
    """
    frame_number = len(frames)
    if frame_number > 6:
        section_number = frame_number // 3
        extra_frames = frame_number % 3
        first_slice = section_number
        second_slice = 2 * section_number + extra_frames
        blue_range = frames[0:first_slice]
        green_range = frames[first_slice:second_slice]
        red_range = frames[second_slice:]
        blue_matrix = calculate_star(blue_range, 1)
        green_matrix = calculate_star(green_range, 2)
        red_matrix = calculate_star(red_range, 3)
        star_representation = combine_matrices(blue_matrix, green_matrix, red_matrix)
        return star_representation

def calculate_star(section, num):
    """
    Calculate Star on one channel of an image
    """
    # print('calculate star')
    h = section[0].shape[0]
    w = section[0].shape[1]
    # print(h, w)
    result = np.zeros((h, w), np.uint8)
    for k in range(1, len(section)):
        # print(section[k].shape)
        # print(section[k])
        current_frame_norm = np.linalg.norm(section[k], axis=2)
        previous_frame_norm = np.linalg.norm(section[k-1], axis=2)
        euclidean = np.absolute(previous_frame_norm - current_frame_norm)
        dot_product = np.vdot(section[k-1], section[k])
        product_of_lengths = current_frame_norm * previous_frame_norm
        angle = 1 - ((dot_product) / (product_of_lengths))
        result = result + (1 - (angle/2)) * euclidean 
    return result

def accumulative_video_motion(frames):
    num_frames = len(frames)
    if num_frames > 0:
        # Backward  
        f = num_frames - 1
        while f > 0:
            key_frame = frames[f]
            if f == num_frames - 1:
                imgDiff = key_frame
            else:
                imgDiff = cv2.addWeighted(imgDiff, 0.5, key_frame, 0.5, 0)
                
            f -= 1

        imgDiff_backward = imgDiff #saving image
            
                # Forward
        f = 0
        while f < num_frames:
            key_frame = frames[f]
            if f == 0:
                imgDiff = key_frame
            else:
                imgDiff = cv2.addWeighted(imgDiff, 0.5, key_frame, 0.5, 0)
            f += 1

        imgDiff_forward = imgDiff
        imgDiff_both = cv2.addWeighted(imgDiff_backward, 0.5, imgDiff_forward, 0.5, 0)
        return imgDiff_both

def combine_matrices(blue, green, red):
    """
    Combine three channels into one image
    """
    image = np.stack((blue, green, red), axis=2)    
    return image

def threaded_function(name):
    """
    This threaded function waits until frames is more than 20
    then calculates the star RGB represenation before
    displaying it. 
    """
    while True:
        time.sleep(0.1)
        global frames
        if len(frames) > 20:
            process_frames = frames
            frames = []
            star_image = accumulative_video_motion(process_frames)
            print(star_image.shape)
            if star_image is not None:
                print('working')
                star_image = cv2.normalize(src=star_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imshow('Processed', cv2.flip(star_image,1))
                cv2.waitKey(1)

"""
We start capture from the connected video device
and save the frames to the frames array and display
the frames in a window
"""
frames = []
cap = cv2.VideoCapture(0)
x = Thread(target=threaded_function, args=(1,))
x.daemon = True
x.start()
while cap.isOpened():
    success, image = cap.read()
    if not success:
      # If loading a video, use 'break' instead of 'continue'.
        continue
    
    frames.append(image)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Video', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()