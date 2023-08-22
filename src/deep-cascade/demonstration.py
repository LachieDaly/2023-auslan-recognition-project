
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

def calculate_width_and_height(border_box):
    width = border_box[2] - border_box[0]
    height = border_box[3] - border_box[1]
    return width, height

def calculate_orientation(width, height):
    value = 0
    if width > height:
        value = 1
    elif height > width:
        value = -1

    return value

def calculate_slope_orientation(left_border_box, right_border_box):

    left_orientation = left_width = left_height = right_orientation = right_width = right_height = slope = 0
    if not left_border_box is None:
        left_width, left_height = calculate_width_and_height(left_border_box)
        left_orientation = calculate_orientation(left_width, left_height)

    if not right_border_box is None:
        right_width, right_height = calculate_width_and_height(right_border_box)
        right_orientation = calculate_orientation(right_width, right_height)

    if not left_border_box is None and not right_border_box is None:
        slope = np.abs((right_border_box[1] + (right_height / 2)) - (left_border_box[1] + (left_height / 2))) / np.abs((right_border_box[0] + (right_width / 2)) - (left_border_box[0] + (left_width / 2)))

    return left_orientation, right_orientation, slope

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

    return np.array([x_min, y_min, x_max, y_max])


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (224, 224))
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    left_border_box = None
    right_border_box = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        for hand in results.multi_handedness:
            info = hand.classification[0]
            index = 0
            if (len(results.multi_handedness) == 2):
                index = info.index
            if (info.label == "Right"):
            # Extract left hand pose
                left_border_box = flatten_key_points(results.multi_hand_landmarks[index])
                
            elif (info.label == "Left"):
                right_border_box = flatten_key_points(results.multi_hand_landmarks[index])

    mask = np.zeros(
        shape=(image.shape[:2]), dtype=np.uint8
    )
    if not left_border_box is None:
        cv2.rectangle(mask, (round(left_border_box[0]*image.shape[1] - 10), round(left_border_box[1]*image.shape[0] - 10)), 
        (round(left_border_box[2]*image.shape[1] + 20), round(left_border_box[3]*image.shape[0] + 20)), 255, -1)

    if not right_border_box is None:
        cv2.rectangle(mask, (round(right_border_box[0]*image.shape[1] - 10), round(right_border_box[1]*image.shape[0] - 10)), 
        (round(right_border_box[2]*image.shape[1] + 20), round(right_border_box[3]*image.shape[0] + 20)), 255, -1)

    left_orientation, right_orientation, slope = calculate_slope_orientation(left_border_box, right_border_box)

    output_image = cv2.bitwise_and(image, image, mask=mask)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    slope_text = "Slope = " + str(slope)
    left_orientation_text = "Left Orientation = " + str(left_orientation)
    right_orientation_text = "Right Orientation = " + str(right_orientation)
    # org
    org = (50, 50)
    org1 = (50, 70)
    org2 = (50, 90)
    # fontScale
    fontScale = 0.5
    # Blue color in BGR
    color = (255, 255, 255)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    # output_image = cv2.flip(output_image, 1)
    output_image = cv2.putText(output_image, slope_text, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    output_image = cv2.putText(output_image, left_orientation_text, org1, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    output_image = cv2.putText(output_image, right_orientation_text, org2, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('Full Image', image)
    cv2.imshow('MediaPipe Hands', output_image)
    if cv2.waitKey(1) & 0xFF == 27:
      break
cap.release()