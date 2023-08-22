
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
TOTAL_FEATURES = CNN_FEATURES + MEDIAPIPE_FEATURES + ESHR_FEATURES
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 10000

train_df = pd.read_csv("C:/Users/Lachie/Desktop/Spreadsheet/train29.csv")
val_df = pd.read_csv("C:/Users/Lachie/Desktop/Spreadsheet/test29.csv")
df = [train_df, val_df]
df = pd.concat(df)
# df = df.tail(10)

# df = df.tail(10)

start_time = time.time()

def crop_center_square(frame):
    """
    
    """
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
    """
    Method to create our hand or full image feature extractor
    frames of videos do not need to be normalised thanks
    to preprocessing the input
    """
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
    """
    Calculates the width and height of a supplied
    border box
    """
    width = border_box[2] - border_box[0]
    height = border_box[3] - border_box[1]
    return width, height

def calculate_orientation(width, height):
    """
    Calculates the orientation of the hand
    based on the width and height supplied
    return feature array of length 25 with
    the calculated orientation
    """
    value = 0
    if width > height:
        value = 1
    elif height > width:
        value = -1

    # print(value)
    return np.full(shape=25, fill_value=value, dtype='int')

def calculate_slope_orientation(left_border_box, right_border_box):
    """
    Calculates absolute slope between left and right border boxes
    uses centre of box
    """
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
    """
    Mediapipe hand landmarks are obtuse to extract
    method to both flatten the 3D keypoints of the hand
    whilst extracting the minimum and maximum x and y values
    for hand bounding boxes
    """
    # Hand have 63 features total
    hand_features = np.zeros(shape=(63), dtype="float32")
    start, end = [0, 3]
    x_min = float('inf')
    y_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')
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

def calculate_x_centre(border_box):
    return border_box[0] + border_box[2] / 2

def get_hand_sides(features, boxes):
    """
    Want to work out handedness in a more consistent fashion than the mediapipe lables
    if arrays contain two elements, handedness is based on how hands are positioned in 
    relation to each other
    If array contains one element, handedness is based on whether the hand is on the left
    hand or right hand side of the screen
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


def extract_eshr_features(left_border_box, right_border_box, image):
    """
    Extract cropped image and orientation and slope using border boxes
    and input image
    """
    mask = np.zeros(
        shape=(image.shape[:2]), dtype=np.uint8
    )

    # Media pipe box is quite tight, increase border box to give some more clues to the pretrained resnet
    # Experimentally we increase this border box by 7 pixels on each side
    if not left_border_box is None:
        cv2.rectangle(mask, (round(left_border_box[0]*IMG_SIZE - 10), round(left_border_box[1]*IMG_SIZE - 10)), 
        (round(left_border_box[2]*IMG_SIZE + 10), round(left_border_box[3]*IMG_SIZE + 10)), 255, -1)

    if not right_border_box is None:
        cv2.rectangle(mask, (round(right_border_box[0]*IMG_SIZE - 10), round(right_border_box[1]*IMG_SIZE) - 10), 
        (round(right_border_box[2]*IMG_SIZE + 10), round(right_border_box[3]*IMG_SIZE + 10)), 255, -1)

    output_image = cv2.bitwise_and(image, image, mask=mask)

    eshr_features = calculate_slope_orientation(left_border_box, right_border_box)

    return output_image, eshr_features

def mediapipe_extraction(frame, prev_left_keypoints, prev_right_keypoints, prev_left_border_box, prev_right_border_box):
    """
    Extract features related to mediapipe including keypoints and ESHR features
    Previous keypoints and border boxes are supplied in case they are not found 
    in the current frame
    """
    mediapipe_features = np.zeros(
        shape=(MEDIAPIPE_FEATURES + ESHR_FEATURES), dtype="float32"
    )

    output_image = np.zeros(
        shape=((IMG_SIZE, IMG_SIZE, 3)), dtype="uint8"
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
        image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        # Work out landmarks
        if results.multi_hand_landmarks:
            index = 0
            for hand in results.multi_handedness:
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
        output_image, mediapipe_features[126:] = extract_eshr_features(left_border_box, right_border_box, image)


    # cv2.imshow("Hands", cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    # cv2.imshow("Full", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # cv2.waitKey(1)
    # print(mediapipe_features)
    return mediapipe_features, output_image, left_border_box, right_border_box


def extract_features(frame, prev_left_keypoints, prev_right_keypoints, left_border_box, right_border_box):
    """
    Method which extracts all features from a given frame
    influenced by previous frames keypoints and border boxes
    """
    features = np.zeros(
        shape=(TOTAL_FEATURES), 
        dtype="float32"
    )

    # Get hand pose features and image with hand cutouts
    features[CNN_FEATURES:], frame, left_border_box, right_border_box = mediapipe_extraction(frame, prev_left_keypoints, prev_right_keypoints, left_border_box, right_border_box)
    features[0:CNN_FEATURES] = feature_extractor.predict(frame[None, ...])
    return features, left_border_box, right_border_box

def prepare_all_videos(df, root_dir):
    """
    Takes our dataframe and video directory and converts
    all videos into their respective feature sequences 
    and labels
    """
    # cv2.namedWindow("Hands")
    # cv2.namedWindow("Full")
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # Frame masks important to tell the model that last frames are unimportant
    # and empty
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, TOTAL_FEATURES), dtype="float32"
    )
    
    # For each video
    for idx, path in enumerate(video_paths):
        frames = load_video(os.path.join(root_dir, path), max_frames=MAX_SEQ_LENGTH)
        temp_frame_features = np.zeros(
            shape=(
                MAX_SEQ_LENGTH, 
                TOTAL_FEATURES
            ), 
            dtype="float32"
        )
        video_length = frames.shape[0]
        left_hand_features = np.zeros(shape=(63), dtype="float32")
        right_hand_features = np.zeros(shape=(63), dtype="float32")
        left_border_box = None
        right_border_box = None
        length = min(MAX_SEQ_LENGTH, video_length)
        for i in range(length):
            temp_frame_features[i, :], left_border_box, right_border_box = extract_features(frames[i],
                left_hand_features, right_hand_features, left_border_box, right_border_box)

            left_hand_features = temp_frame_features[i, :][512:575]
            right_hand_features = temp_frame_features[i, :][575:638]

        # No real use of a mask here anymore
        # frame_features

        frame_features[
            idx,
        ] = temp_frame_features

    return frame_features, labels


data, labels = prepare_all_videos(df, "C:/Users/Lachie/Desktop/Reconstructed")
np.save('./consistent-hands/data.npy', data, allow_pickle=True)
# np.save('./recognition-before-resize/train_masks.npy', data[1], allow_pickle=True)
np.save('./consistent-hands/labels.npy', labels, allow_pickle=True)
