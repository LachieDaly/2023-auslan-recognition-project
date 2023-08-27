
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
MAX_SEQ_LENGTH = 20
CNN_FEATURES = 512
MEDIAPIPE_FEATURES = 126
TOTAL_FEATURES = CNN_FEATURES + MEDIAPIPE_FEATURES
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 1000

train_df = pd.read_csv("C:/Users/Lachie/Desktop/Spreadsheet/new_train.csv")
val_df = pd.read_csv("C:/Users/Lachie/Desktop/Spreadsheet/new_val.csv")
start_time = time.time()

seq = va.Sequential([
    va.RandomCrop(size=(IMG_SIZE, IMG_SIZE)),
    va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
    sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
])

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor():
    feature_extractor = keras.applications.ResNet50V2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    # Required for resnet trained on imagenet
    preprocess_input = keras.applications.resnet_v2.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    outputs = keras.layers.Dense(CNN_FEATURES)(outputs)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(val_df["tag"])
)
print(label_processor.get_vocabulary())

def flatten_key_points(hand_landmarks):
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

def create_hand_cutouts(left_box_borders, right_box_borders, image):
    mask = np.zeros(
        shape=(image.shape[:2]), dtype=np.uint8
    )

    if not left_box_borders is None:
        cv2.rectangle(mask, (round(left_box_borders[0]*IMG_SIZE), round(left_box_borders[1]*IMG_SIZE)), (round(left_box_borders[2]*IMG_SIZE), round(left_box_borders[3]*IMG_SIZE)), 255, -1)

    if not right_box_borders is None:
        cv2.rectangle(mask, (round(right_box_borders[0]*IMG_SIZE), round(right_box_borders[1]*IMG_SIZE)), (round(right_box_borders[2]*IMG_SIZE), round(right_box_borders[3]*IMG_SIZE)), 255, -1)

    output_image = cv2.bitwise_and(image, image, mask=mask)
    return output_image

def mediapipe_extraction(batch_frame, prev_left_keypoints, prev_right_keypoints):
    mediapipe_features = np.zeros(
        shape=(MEDIAPIPE_FEATURES), dtype="float32"
    )

    image = batch_frame[0]
    
    output_image = np.zeros(
        shape=(image.shape), dtype="float32"
    )
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.1) as hands:

        # Where the magic happens 
        results = hands.process(image)

        # Work out landmarks
        if results.multi_hand_landmarks:
            left_hand_detected = False
            left_box_borders = None
            right_hand_detected = False
            right_box_borders = None
            for hand in results.multi_handedness:
                info = hand.classification[0]
                index = 0
                if (len(results.multi_handedness) == 2):
                    index = info.index
                if (info.label == "Left"):
                    # Extract left hand pose
                    left_hand_detected = True
                    mediapipe_features[0:63], left_box_borders = flatten_key_points(results.multi_hand_landmarks[index])
                elif (info.label == "Right"):
                    # Extract right hand pose
                    right_hand_detected = True
                    mediapipe_features[63:126], right_box_borders = flatten_key_points(results.multi_hand_landmarks[index])

                output_image = create_hand_cutouts(left_box_borders, right_box_borders, image)

            if not left_hand_detected:
                mediapipe_features[0:63] = prev_left_keypoints

            if not right_hand_detected:
                mediapipe_features[63:126] = prev_right_keypoints

        cv2.imshow("Hands", image)

    return mediapipe_features, output_image


def extract_features(batch_frame, prev_left_keypoints, prev_right_keypoints):
    # Create feature array
    features = np.zeros(
        shape=(TOTAL_FEATURES), 
        dtype="float32"
    )

    # Get hand pose features and image with hand cutouts
    features[CNN_FEATURES:], batch_frame[0] = mediapipe_extraction(batch_frame, prev_left_keypoints, prev_right_keypoints)

    features[0:CNN_FEATURES] = feature_extractor.predict(batch_frame, verbose=0)

    return features

def prepare_all_videos(df, root_dir, augment):
    cv2.namedWindow("Hands")
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
        frames = load_video(os.path.join(root_dir, path))
        if augment:
            seq(frames)
        frames = frames[None, ...]

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
            for j in range(length):
                temp_frame_features[i, j, :] = extract_features(batch[None, j, :], left_hand_features, right_hand_features)
                left_hand_features = temp_frame_features[i, j, :][512:575]
                right_hand_features = temp_frame_features[i, j, :][575:]
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[
            idx,
        ] = temp_frame_features.squeeze()
        frame_masks[
            idx,
        ] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels

val_data, val_labels = prepare_all_videos(val_df, "C:/Users/Lachie/Desktop/Reconstructed", False)

# Utility for our sequence model.
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, TOTAL_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    x = keras.layers.LSTM(25, kernel_regularizer=keras.regularizers.L2(0.0005))(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(25, kernel_regularizer=keras.regularizers.L2(0.0005), activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)
    rnn_model = keras.Model([frame_features_input, mask_input], output)

    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    return rnn_model

# Utility for running experiments.
def run_experiment():
    filepath = "./tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    train_acc = []
    val_acc = []
    seq_model = get_sequence_model()
    # Not viable for thesis
    for i in range(EPOCHS):
        train_data, train_labels = prepare_all_videos(train_df, "C:/Users/Lachie/Desktop/Reconstructed", True)
        history = seq_model.fit(
            [train_data[0], train_data[1]],
            train_labels,
            batch_size=BATCH_SIZE,
            validation_data=([val_data[0], val_data[1]], val_labels),
            epochs=1,
            callbacks=[checkpoint],
        )
        train_acc.append(history.history['accuracy'])
        val_acc.append(history.history['val_accuracy'])
    end_time = time.time()
    print("Time")
    print(end_time - start_time)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend(['accuracy', 'validation accuracy'])
    plt.show()
    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([val_data[0], val_data[1]], val_labels)
    predictions_one_hot = seq_model.predict([val_data[0], val_data[1]])
    print(val_labels)
    print(predictions_one_hot)
    cm = confusion_matrix(val_labels.argmax(axis=1), predictions_one_hot.argmax(axis=1))
    print(cm)
    classes = ('Arrive', 'Bad', 'Bed', 'Bird', 'Boy', 'Come', 'Day', 'Eat', 'Frog', 'Good', 'House', 'Laugh', 'Man', 'Night', 'People', 'Real', 'Slow', 'Sprint', 'Think', 'What', 'Where', 'Window')
    df_cm = pd.DataFrame(cm/np.sum(cm) * 10, index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('media-pipe-lstm.png')
    # seq_model.predict([X_val, X_val_mask], y_val)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()