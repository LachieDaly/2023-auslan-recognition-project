"""
https://github.com/Hamzah-Luqman/SLR_AMN/blob/main/frame.py

Extract frames from a video (or many videos).
Plust some frame=image manipulation utilities
"""

import os
import glob
import warnings
import random
from subprocess import check_output
import re

import numpy as np
import pandas as pd

import cv2

def resize_image(image:np.array, min_dim:int=256) -> np.array:
    height, width, _ = image.shape

    if width >= height:
        ratio = min_dim / height
    else:
        ratio = min_dim / width
    
    if ratio != 1.0:
        image = cv2.resize(image, dsize=(0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)

    return image

def resize_images(images:np.array, min_dim:int=256) -> np.array:
    num_images, _, _, _ = images.shape
    image_list = []
    for i in range(num_images):
        image = resize_image(images[i, ...])
        image_list.append(image)
    return np.array(image_list)

def resize_images_v2(images:np.array, min_dim:int=256) -> np.array:
    num_images, _, _, _ = images.shape
    image_list = []
    for i in range(num_images):
        image = resize_image(images[i, ...], dsize=(min_dim,min_dim))
        image_list.append(image)
    return np.array(image_list)

def resize_images_v3(images:np.array, min_dim:int=256) -> np.array:
    num_images, _, _, _ = images.shape
    image_list = []
    for i in range(num_images):
        image = cv2.resize(images[i, ...], dsize=(min_dim,min_dim))
        image_list.append(image)
    return np.array(image_list)


def rgb_to_gray(images:np.array, min_dim:int=256) -> np.array:
    num_images, _, _, _ = images.shape
    image_list = []
    for i in range(num_images):
        image = cv2.resize(images[i, ...], cv2.COLOR_BGR2GRAY)
        image_list.append(image)
    return np.array(image_list)

def video_to_frames(video_path:str, resize_min_dim:int) -> np.array:
    """
    Read video file with OpenCV and return array of frames
    The frame rate depends on the video (and cannot be set)

    if resize_min_dim != None: frames are resized preserving aspect ratio
    so that the smallest dimension is eg 256 pixels, with bilinear interpolation
    """
    video = cv2.VideoCapture(video_path)
    if (video.isOpened() == False):
        raise ValueError("Error opening video file")

    frame_list = []

    # Read until video is completed
    while True:
        (frame_grabbed, frame) = video.read()

        if frame_grabbed == False:
            break

        if resize_min_dim != None:
            frame_resized = resize_image(frame, resize_min_dim)

        frame_list.append(frame_resized)

    return np.array(frame_list)

def frames_to_files(frames:np.array, target_dir:str):
    """
    Write array of frames to png files
    Input: frames = (number of frames, height, width, depth)
    """
    os.makedirs(target_dir, exist_ok=True)
    for frame in range(frames.shape[0]):
        cv2.imwrite(target_dir + "/frame%04d.png" % frame, frames[frame, :, :, :])
    return

def files_to_frames(path:str, convert_to_graph:bool=False, resize_img:bool=False) -> np.array:
    files = sorted(glob.glob(path + "/*.png"))
    if len(files) == 0:
        raise ValueError("No frames found in " + path)

    frames = []

    for frame_path in files:
        frame = cv2.imread(frame_path)
        frames.append(frame)

    return np.array(frames)


def frames_downsample(frames:np.array, frames_target:int) -> np.array:
    """
    Adjust number of frames (eg 123) to frames_target (eg 79)
    works also if originally less frames then frames_target
    """
    print(frames.shape, frames_target)
    sample_num, _, _, _ = frames.shape
    if sample_num == frames_target:
        return frames
    
    fraction = sample_num / frames_target
    index = [int(fraction * i) for i in range(frames_target)]
    target = [frames[i,:,:,:] for i in index]

    return np.array(target)

def crop_image(frame, height_target, width_target) -> np.array:
    """
    Crop 1 frame to specified size, choose centered image
    """
    height, width, _ = frame.shape

    if (height < height_target) or (width < width_target):
        raise ValueError("Image height/width too small to crop to target size")

    x = int(width/2 - width_target/2)
    y = int(height/2 - height_target/2)

    frame = frame[y:y+height_target, x:x+width_target, :]

    return frame

def crop_images(frames:np.array, height_target, width_target) -> np.array:
    """
    Crop each frame in array to specified size, choose centered image
    """
    sample_num, height, width, depth = frames.shape

    if (height < height_target) or (width < width_target):
        frames = resize_images(frames, height_target)
    else:
        x = int(width/2 - width_target/2)
        y = int(height/2 - height_target/2)

        frames = frames[:, y:y+height_target, x:x+width_target, :]

    return frames

def normalise_images(frames:np.array) -> np.array(float):
    """
    Rescale array of images (rgb 0-255) to [-1.0, 1.0]
    """

    normalised_frames = frames / 127.5
    normalised_frames -= 1.

    return normalised_frames

def process_images(frames:np.array, frame_num:int, height:int, width:int, rescale:bool=True) -> np.array(float):
    """
    All processing
        - downsample number of frames
        - crop to centered image
        - rescale rgb 0-255 value to [-1.0, 1.0] - only if rescale == True
    """
    frames = frames_downsample(frames, frame_num)

    frames = crop_images(frames, height, width)

    if rescale:
        frames = normalise_images(frames)
    elif np.max(np.abs(frames)) > 1.0:
        warnings.warn("images not normalised")

    return frames

def images_normalize_withGrayscale(frames:np.array, frame_num:int, height:int, width:int, rescale:bool = True) -> np.array(float):
    """ Several image normalizations/preprocessing: 
        - downsample number of frames
        - crop to centered image
        - rescale rgb 0-255 value to [-1.0, 1.0] - only if bRescale == True
		- Convert to gray and resize
		- Convert it tp 1-D
    Returns array of floats
    """

    # normalize the number of frames (assuming typically downsampling)
    frames = frames_downsample(frames, frame_num)

	#Hamzah: convert to grayscale
    frames = rgb_to_gray(frames, height, width)
	
    # crop to centered image
    frames = crop_images(frames, height, width)

    if rescale:
        # normalize to [-1.0, 1.0]
        frames = normalise_images(frames)
    elif np.max(np.abs(frames)) > 1.0: 
        warnings.warn("Images not normalized")

    return frames

def frames_show(frames:np.array, wait_milliseconds:int = 100):

    frames_count, nHeight, nWidth, nDepth = frames.shape
    
    for i in range(frames_count):
        cv2.imshow("Frame", frames[i, :, :, :])
        cv2.waitKey(wait_milliseconds)

    return

def video_length(video_path:str) -> float:
    print(video_path)
    return int(check_output(['mediainfo', '--Inform=Video;%Duration%', video_path]))/1000.0

def video_dir_to_frames_dir(video_dir:str, frame_dir:str, frames_norm:int = None, 
    resize_min_dim:int = None, crop_shape:tuple = None, classes:int = None):
    """ Extract frames from videos 
    
    Input video structure:
    ... video_dir / train / class001 / videoname.avi
    Output:
    ... frame_dir / train / class001 / videoname / frames.png
    """

    # do not (partially) overwrite existing frame directory
    #if os.path.exists(frame_dir): 
    #    warnings.warn("Frame folder " + frame_dir + " already exists, frame extraction stopped")
    #    return 

    # get videos. Assume video_dir / train / class / video.mp4
    videos_df = pd.DataFrame(sorted(glob.glob(video_dir + "/*/*/*.*")), columns=["video_path"])
    print("Located {} videos in {}, extracting to {} ..."\
        .format(len(videos_df), video_dir, frame_dir))
    if len(videos_df) == 0: raise ValueError("No videos found")

    # eventually restrict to first nLabels
    if classes != None:
        videos_df.loc[:,"label"] = videos_df.video_path.apply(lambda s: s.split("/")[-2])
        classes = sorted(videos_df.label.unique())[:classes]
        videos_df = videos_df[videos_df["label"].isin(classes)]
        print("Using only {} videos from first {} classes".format(len(videos_df), classes))

    counter = 0
    # loop through all videos and extract frames
    for video_path in videos_df.video_path:

        # assemble target diretory (assumed directories see above)
        video_path = os.path.normpath(video_path)
        li_video_path = video_path.split("\\")
        if len(li_video_path) < 4: raise ValueError("Video path should have min 4 components: {}".format(str(li_video_path)))
        video_name = li_video_path[-1].split(".")[0]
        target_dir = frame_dir + "/" + li_video_path[-3] + "/" + li_video_path[-2] + "/" + video_name
        
        # check if frames already extracted
        if frames_norm != None and os.path.exists(target_dir):
            frames_count = len(glob.glob(target_dir + "/*.*"))
            if frames_count == frames_norm: 
                print("Video %5d already extracted to %s" % (counter, target_dir))
                counter += 1
                continue
            else: 
                print("Video %5d: Directory with %d instead of %d frames detected" % (counter, frames_count, frames_norm))
        
        # create target directory
        os.makedirs(target_dir, exist_ok = True)

        # slice videos into frames with OpenCV
        frames = video_to_frames(video_path, resize_min_dim)

        # length and fps
        # video_sec = video_length(video_path)
        frames_count = len(frames)
        # fps = frames_count / video_sec   

        fps = 25;

        # downsample
        if frames_norm != None:
            print(video_path)
            frames = frames_downsample(frames, frames_norm)

        # crop images
        if crop_shape != None:
            frames = crop_images(frames, *crop_shape)
        
        # write frames to .png files
        frames_to_files(frames, target_dir)         

        print("Video %5d | %d frames | %4.1f fps | saved %s in %s" % (counter, frames_count, fps, str(frames.shape), target_dir))
        counter += 1      

    return


def unittest(video_dir, nSamples = 100):
    print("\nAnalyze video durations and fps from %s ..." % (video_dir))
    print(os.getcwd())

    videos = glob.glob(video_dir + "/*/*.mp4") + glob.glob(video_dir + "/*/*.avi")
    
    if len(videos) == 0: raise ValueError("No videos detected")

    video_sec_sum, frames_count_sum = 0, 0
    for i in range(nSamples):
        video_path = random.choice(videos)
        #print("Video %s" % video_path)

        # read video
        frames = video_to_frames(video_path, 256)
        frames_count = len(frames)

        # determine length of video in sec and deduce frame rate
        video_sec = video_length(video_path)
        fps = frames_count / video_sec

        video_sec_sum += video_sec
        frames_count_sum += frames_count

        print("%2d: Shape %s, duration %.1f sec, fps %.1f" % (i, str(frames.shape), video_sec, fps))

    count = i+1
    print("%d samples: Average video duration %.1f sec, fps %.1f" % (nSamples, video_sec_sum / count, frames_count_sum / video_sec_sum))

    return
