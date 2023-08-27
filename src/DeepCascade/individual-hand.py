
import numpy as np
import sys
import tensorflow as tf
import os
from datetime import datetime
import seaborn as sn
import cv2
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
import time
from numpy import asarray
from numpy import savetxt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from collections import defaultdict

IMG_SIZE = 224
IMG_RESIZE_SIZE = IMG_SIZE + 15
MAX_SEQ_LENGTH = 20
CNN_FEATURES = 512
MEDIAPIPE_FEATURES = 126
ESHR_FEATURES = 75
TOTAL_FEATURES = 2*CNN_FEATURES + MEDIAPIPE_FEATURES + ESHR_FEATURES
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 10000

train_df = pd.read_csv("C:/Users/Lachie/Desktop/Spreadsheet/train29.csv")
val_df = pd.read_csv("C:/Users/Lachie/Desktop/Spreadsheet/test29.csv")
df = [train_df, val_df]
df = pd.concat(df)
print(df.shape)

# df = df.head(10)

start_time = time.time()


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=0):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = crop_center_square(frame)

            # Flip to RGB
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()

    last_frame = frames[-1]
    while len(frames) < max_frames:
        frames.append(last_frame)

    return np.array(frames)

def build_feature_extractor():
    feature_extractor = keras.applications.vgg16.VGG16(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    # Required for preprocessing trained on imagenet
    preprocess_input = keras.applications.vgg16.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    # print(outputs.shape)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
)
print(label_processor.get_vocabulary())

def calculate_width_and_height(border_box):
    width = border_box[2] - border_box[0]
    height = border_box[3] - border_box[1]
    return width, height

def calculate_x_centre(border_box):
    return border_box[0] + border_box[2] / 2

def calculate_orientation(width, height):
    value = 0
    if width > height:
        value = 1
    elif height > width:
        value = -1

    # print(value)
    return np.full(shape=25, fill_value=value, dtype='int')

def calculate_slope_orientation(left_border_box, right_border_box):
    eshr_features = np.zeros(
        shape=(ESHR_FEATURES), dtype="float32"
    )
    left_width = left_height = right_width = right_height = slope = 0
    if not left_border_box is None:
        left_width, left_height = calculate_width_and_height(left_border_box)
        eshr_features[:25] = calculate_orientation(left_width, left_height)

    if not right_border_box is None:
        right_width, right_height = calculate_width_and_height(right_border_box)
        eshr_features[25:50] = calculate_orientation(right_width, right_height)

    if not left_border_box is None and not right_border_box is None:
        slope = np.abs((right_border_box[1] + (right_height / 2)) - (left_border_box[1] + (left_height / 2))) / np.abs((right_border_box[0] + (right_width / 2)) - (left_border_box[0] + (left_width / 2)))
    
    eshr_features[50:75] = np.full(shape=25, fill_value=slope, dtype="float32")

    return eshr_features


def flatten_key_points(hand_landmarks):
    # Hand have 63 features total
    hand_features = np.zeros(shape=(63), dtype="float32")

    start, end = [0, 3]
    x_min = float('inf')
    y_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')
    # Future improvements!!!!
    for hand_landmark in hand_landmarks.landmark:

        if x_min > hand_landmark.x:
            x_min = hand_landmark.x
        
        if x_max < hand_landmark.x:
            x_max = hand_landmark.x

        if y_min > hand_landmark.y:
            y_min = hand_landmark.y

        if y_max < hand_landmark.y:
            y_max = hand_landmark.y

        hand_features[start:end] = np.array([hand_landmark.x, hand_landmark.y, hand_landmark.z])
        start += 3
        end += 3

    return hand_features, np.array([x_min, y_min, x_max, y_max])

def get_hand_cutout(border_box, image):
    # Extra pixels around border box are arbitrary and experimental
    img_y, img_x = image.shape[0:2]


    if not border_box is None:
        # print(border_box)
        x_min = round(border_box[0]*img_x - 10)
        y_min = round(border_box[1]*img_y - 10)
        x_max = round(border_box[2]*img_x + 10)
        y_max = round(border_box[3]*img_y + 10)
        hand_image = image[y_min:y_max, x_min:x_max]
        # print(hand_image.shape)
        try:
            return cv2.resize(hand_image, (IMG_SIZE, IMG_SIZE))
        except:
            print('could not resize')

    return np.zeros(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)


def extract_eshr_features(left_border_box, right_border_box, image):

    # Media pipe box is quite tight, increase border box to give some more clues to the pretrained resnet
    # Experimentally we increase this border box by 7 pixels on each side
    
    left_image = get_hand_cutout(left_border_box, image)
    right_image = get_hand_cutout(right_border_box, image)
    eshr_features = calculate_slope_orientation(left_border_box, right_border_box)
    return left_image, right_image, eshr_features

def get_hand_sides(features, boxes):
    """
    Want to work out handedness in a more consistent fashion
    Hands crossing over will change the features however
    """
    left_features = right_features = np.zeros(
        shape=(63), dtype="float32"
    )
    left_box = right_box = None
    # If two hands are detected decide handedness based on both positions
    # Hard if two hands happen to cross over, which is absolutely a possibility
    if len(features) == 2:
        box_1_x = calculate_x_centre(boxes[0])
        box_2_x = calculate_x_centre(boxes[1])
        # Doesn't matter how we distribute left or right hand as long as it is consistent
        # Not super happy with this implementation
        if (box_1_x >= box_2_x):
            # Give box_1_x right hand (not necessarily right hand, but right side of screen)
            right_features = features[0]
            right_box = boxes[0]
            # Give box_2_x left hand (not necessarily left hand, but left side of screen)
            left_features = features[1]
            left_box = boxes[1]
        else:
            right_features = features[1]
            right_box = boxes[1]
            left_features = features[0]
            left_box = boxes[0]            

    # If one hand is detected, determine based on side of image
    elif len(features) == 1:
        box_x = calculate_x_centre(boxes[0])
        if box_x >= 0.5:
            right_features = features[0]
            right_box = boxes[0]
        else:
            left_features = features[0]
            left_box = boxes[0]

    keypoints = np.concatenate((left_features, right_features))

    return keypoints, left_box, right_box

def mediapipe_extraction(frame, prev_left_keypoints, prev_right_keypoints, prev_left_border_box, prev_right_border_box):
    mediapipe_features = np.zeros(
        shape=(MEDIAPIPE_FEATURES + ESHR_FEATURES), dtype="float32"
    )
    features = []
    boxes = []
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.01) as hands:
        # Where the magic happens 
        # print(frame.shape)
        results = hands.process(frame)
        # immediately resize frame
        # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        # Work out landmarks

        if results.multi_hand_landmarks:
            index = 0
            for hand in results.multi_handedness:
                # Extract left hand pose
                feature, border_box = flatten_key_points(results.multi_hand_landmarks[index])
                features.append(feature)
                boxes.append(border_box)
                index += 1


        mediapipe_features[:126], left_border_box, right_border_box = get_hand_sides(features, boxes)

        if left_border_box is None:
            mediapipe_features[:63] = prev_left_keypoints
            left_border_box = prev_left_border_box

        if right_border_box is None:
            mediapipe_features[63:126] = prev_right_keypoints
            right_border_box = prev_right_border_box

        # Crop image and extract orientation and slope based on border boxes
        # border boxes may be from previous frame
        left_hand_image, right_hand_image, mediapipe_features[126:] = extract_eshr_features(left_border_box, right_border_box, frame)
    return mediapipe_features, left_hand_image, right_hand_image, left_border_box, right_border_box


def extract_features(frame, prev_left_keypoints, prev_right_keypoints, left_border_box, right_border_box):
    # Create feature array
    features = np.zeros(
        shape=(TOTAL_FEATURES), 
        dtype="float32"
    )

    # Get hand pose features and image with hand cutouts
    # Extracted media pipe features go after CNN extracted features
    features[2*CNN_FEATURES:], left_hand_frame, right_hand_frame, left_border_box, right_border_box = mediapipe_extraction(frame, prev_left_keypoints, prev_right_keypoints, left_border_box, right_border_box)
    # Left hand cnn features
    features[0:CNN_FEATURES] = feature_extractor.predict(left_hand_frame[None, ...], verbose=0)
    # right hand cnn features
    features[CNN_FEATURES:2*CNN_FEATURES] = feature_extractor.predict(right_hand_frame[None, ...], verbose=0)
    return features, left_border_box, right_border_box

def prepare_all_videos(df, root_dir):
    # cv2.namedWindow("Left Hand")
    # cv2.namedWindow("Right Hand")
    # cv2.namedWindow("Full")
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # Frame masks important to tell the model that last frames are unimportant
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, TOTAL_FEATURES), dtype="float32"
    )
    
    # For each video
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path), max_frames=MAX_SEQ_LENGTH)
        # frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.

        temp_frame_features = np.zeros(
            shape=(
                MAX_SEQ_LENGTH, 
                TOTAL_FEATURES
            ), 
            dtype="float32"
        )
        # print(temp_frame_features.shape)
        video_length = frames.shape[0]
        left_hand_features = np.zeros(shape=(63), dtype="float32")
        right_hand_features = np.zeros(shape=(63), dtype="float32")
        left_border_box = None
        right_border_box = None
        length = min(MAX_SEQ_LENGTH, video_length)
        for i in range(length):
            temp_frame_features[i, :], left_border_box, right_border_box = extract_features(frames[i],
                left_hand_features, right_hand_features, left_border_box, right_border_box)

            # Pass features from last frame
            left_hand_features = temp_frame_features[i, :][2*CNN_FEATURES:2*CNN_FEATURES+63]
            right_hand_features = temp_frame_features[i, :][2*CNN_FEATURES+63:2*CNN_FEATURES+126]

        # No real use of a mask here anymore
        # frame_features

        frame_features[
            idx,
        ] = temp_frame_features

    return frame_features, labels


data, labels = prepare_all_videos(df, "C:/Users/Lachie/Desktop/Reconstructed")
np.save('./individual-hand/data.npy', data, allow_pickle=True)
# np.save('./recognition-before-resize/train_masks.npy', data[1], allow_pickle=True)
np.save('./individual-hand/labels.npy', labels, allow_pickle=True)
