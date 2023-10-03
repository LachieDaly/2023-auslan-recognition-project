
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

from vidaug import augmentors as va

sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability

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

seq = va.Sequential([
    va.RandomCrop(size=(IMG_SIZE, IMG_SIZE)),
    va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
    sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
])

train_df = pd.read_csv("C:/Users/Lachie/Desktop/Spreadsheet/train29.csv")
train_df = train_df.tail(1)
# val_df = pd.read_csv("C:/Users/Lachie/Desktop/Spreadsheet/test29.csv")
start_time = time.time()


def crop_center_square(frame):
    """
    Crops a rectangular image into a square
    """
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, augment=False, resize=(IMG_SIZE, IMG_SIZE)):
    """
    Load up to the maximum number of frames into a numpy array
    augment/resize commented out
    """
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = crop_center_square(frame)

            # if not augment:
            #     frame = cv2.resize(frame, resize)
            # else:
            #     frame = cv2.resize(frame, (IMG_RESIZE_SIZE, IMG_RESIZE_SIZE))
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

    if augment:
        frames = seq(frames)
    return np.array(frames)

def build_feature_extractor():
    """
    Returns a pretrained VGG16 convolutional model feature extractor
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
    Calculate width and height values of supplied box
    """
    width = border_box[2] - border_box[0]
    height = border_box[3] - border_box[1]
    return width, height

def calculate_orientation(width, height):
    """Calculate orientation value given width and height of a box"""
    value = 0
    if width > height:
        value = 1
    elif height > width:
        value = -1

    # print(value)
    return np.full(shape=25, fill_value=value, dtype='int')

def calculate_slope_orientation(left_border_box, right_border_box):
    """
    Calculate and return slope and orientation features based on border boxes
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
    Flattent and return the hand landmarks returned by mediapipe hands 
    in addition to the bounding box for the hand
    """
    # Hands have 63 features total
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

def extract_eshr_features(left_border_box, right_border_box, image):
    """
    Calculates and returns accessory features and masked image based on border boxes
    """
    mask = np.zeros(
        shape=(image.shape[:2]), dtype=np.uint8
    )

    # Media pipe box is quite tight, increase border box to give some more clues to the pretrained resnet
    if not left_border_box is None:
        cv2.rectangle(mask, (round(left_border_box[0]*image.shape[1] - 10), round(left_border_box[1]*image.shape[0] - 10)), 
        (round(left_border_box[2]*image.shape[1] + 10), round(left_border_box[3]*image.shape[0] + 10)), 255, -1)

    if not right_border_box is None:
        cv2.rectangle(mask, (round(right_border_box[0]*image.shape[1] - 10), round(right_border_box[1]*image.shape[0] - 10)), 
        (round(right_border_box[2]*image.shape[1] + 10), round(right_border_box[3]*image.shape[0] + 10)), 255, -1)


    output_image = cv2.bitwise_and(image, image, mask=mask)

    eshr_features = calculate_slope_orientation(left_border_box, right_border_box)

    return output_image, eshr_features

def mediapipe_extraction(batch_frame, prev_left_keypoints, prev_right_keypoints, left_border_box, right_border_box, frame_no=0):
    """
    Pass image through mediapipe hands for keypoint extraction, and perform all necessary calculations from these keypoints
    """
    mediapipe_features = np.zeros(
        shape=(MEDIAPIPE_FEATURES + ESHR_FEATURES), dtype="float32"
    )

    image = cv2.normalize(src=batch_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    output_image = np.zeros(
        shape=((IMG_SIZE, IMG_SIZE, 3)), dtype="uint8"
    )
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        # Where the magic happens 
        results = hands.process(image)

        # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        # Work out landmarks
        if results.multi_hand_landmarks:
            left_hand_detected = False
            right_hand_detected = False
            for hand in results.multi_handedness:
                info = hand.classification[0]
                index = 0
                # print(results.multi_handedness)
                if (len(results.multi_handedness) == 2):
                    index = info.index
                if (info.label == "Left"):
                    # Extract left hand pose
                    left_hand_detected = True
                    mediapipe_features[0:63], left_border_box = flatten_key_points(results.multi_hand_landmarks[index])
                elif (info.label == "Right"):
                    # Extract right hand pose
                    right_hand_detected = True
                    mediapipe_features[63:126], right_border_box = flatten_key_points(results.multi_hand_landmarks[index])

                output_image, mediapipe_features[126:] = extract_eshr_features(left_border_box, right_border_box, image)

            if not left_hand_detected:
                mediapipe_features[0:63] = prev_left_keypoints

            if not right_hand_detected:
                mediapipe_features[63:126] = prev_right_keypoints

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
    print(batch_frame)
    cv2.imshow("Hands", cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    cv2.imshow("Full", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    frame_number = str(frame_no).zfill(4)
    cv2.imwrite("./images/output_%s.jpg" % frame_number, cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    cv2.imwrite("./images/input_%s.jpg" % frame_number, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)
    return mediapipe_features, output_image, left_border_box, right_border_box


def extract_features(frame, prev_left_keypoints, prev_right_keypoints, left_border_box, right_border_box, frame_no=0):
    """
    Extract hand spatial features from the masked image
    """
    # Create feature array
    features = np.zeros(
        shape=(TOTAL_FEATURES), 
        dtype="float32"
    )

    # Get hand pose features and image with hand cutouts
    features[CNN_FEATURES:], frame, left_border_box, right_border_box = mediapipe_extraction(frame, prev_left_keypoints, prev_right_keypoints, left_border_box, right_border_box, frame_no=frame_no)

    # features[0:CNN_FEATURES] = feature_extractor.predict(batch_frame)

    return features, left_border_box, right_border_box

def prepare_all_videos(df, root_dir, augment):
    """
    Turn all videos into features
    """
    # cv2.namedWindow("Hands")
    # cv2.namedWindow("Full")
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # Frame masks important to tell the model that last frames are unimportant
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, TOTAL_FEATURES), dtype="float32"
    )
    
    # For each video
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path), augment=augment, max_frames=MAX_SEQ_LENGTH)

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(
            shape=(
                1,
                MAX_SEQ_LENGTH,
            ),
            dtype="bool",
        )
        temp_frame_features = np.zeros(
            shape=(
                1, 
                MAX_SEQ_LENGTH, 
                TOTAL_FEATURES
            ), 
            dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            left_hand_features = np.zeros(shape=(63), dtype="float32")
            right_hand_features = np.zeros(shape=(63), dtype="float32")
            left_border_box = None
            right_border_box = None
            for j in range(length):
                # temp_frame_features[i, j, :], left_border_box, right_border_box = 
                extract_features(frames[j], left_hand_features, right_hand_features, left_border_box, right_border_box, frame_no=j)

                left_hand_features = temp_frame_features[i, j, :][512:575]
                right_hand_features = temp_frame_features[i, j, :][575:638]
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[
            idx,
        ] = temp_frame_features.squeeze()
        frame_masks[
            idx,
        ] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


train_data, train_labels = prepare_all_videos(train_df, "C:/Users/Lachie/Desktop/Reconstructed", False)
