"""
This code has been used with some moficiations from
https://github.com/Hamzah-Luqman/SLR_AMN/blob/main/prepareData.py

Assume videos are stored in sVideoDir as:
... sVideoDir / train / class001 / gesture.mp4
... sVideoDir / val / class249 / gesture.avi

This pipeline
* extracts frames/images from videos (saved in sVideoDir path) and save frames in 
* Extract MobileNet features from frames and save them in diFeature folder
"""

import os
from frame import video_dir_to_frames_dir
from feature import load_feature_extractor, predict_features

def start_pipeline(video_set, video_dir, feature, image_dir, image_feature_dir):
    # Extract frames from videos
    video_dir_to_frames_dir(video_dir, image_dir, frames_norm=video_set["frames_norm"], 
                            resize_min_dim=video_set["min_dim"], crop_shape=feature["input_shape"][0:2])

    # Load pretrained MobileNet model without top layer
    model = load_feature_extractor(feature)

    # Calculate MobileNet features from rgb frames
    predict_features(image_dir + "/test", image_feature_dir + "/test", model, video_set["frames_norm"])

    predict_features(image_dir + "/train", image_feature_dir + "/train", model, video_set["frames_norm"])


# dataset
video_set = {
    "name": "ELAR",
    "classes": 30, # number of classes
    "frames_norm": 18, # number of frames per video
    "min_dim": 224, # Smaller dimension of saved video frames
    "shape": (224, 224), # height, width
    "fps_avg": 30,
    "frames_avg": 18,
    "duration_avg": 4.0
}

# feature extractor
feature = {
    "name": "mobilenet",
    "input_shape": (224, 224, 3),
    "output_shape": (1024, )
}

video_dir = '../dataset/elar/videos/' # path of video files
image_dir = '../dataset/elar/images/' # path of image files

image_feature_dir = './dataset/elar/features/mobilenet_temp/25/images/01'

print("Extracting frames, optical flow and ... features ...")
print(os.getcwd())

start_pipeline(video_set, video_dir, feature, image_dir, image_feature_dir)