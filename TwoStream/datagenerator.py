"""
https://github.com/Hamzah-Luqman/SLR_AMN/blob/main/datagenerator.py

For neural network training the method Keras.model.fit_generator is used
this requires a generatro that reads and yields training data to the Keras engine.
"""

import glob
import os
import sys
import cv2

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import keras
import tensorflow as tf
from keras.utils import np_utils
