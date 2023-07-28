"""
https://github.com/Hamzah-Luqman/SLR_AMN/blob/main/feature.py

In some video classification NN architectures it may be necessary to calculate
features from the (video) frames, that are afterwards used for NN training

Eg in the MobileNet-LSTM architecture, the video frames are first fed into the
MobileNet and the resulting 1024 **features** saved to disc.
"""

import os
import glob
import time
import sys
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.xception import Xception
from keras.applications.efficientnet import EfficientNetB0
from keras import layers

from keras.layers import Dense # Fully connected layer

from datagenerator import FramesGenerator
from keras.layers import Flatten, Dropout
from keras.models import Sequential
from keras.layers.convolutional import (Conv1D, Conv2D, MaxPooling2D)

def load_feature_extractor(di_feature:dict) -> keras.Model:

    model_name = di_feature["name"]
    print("Load 2D extraction model %s ..." % model_name)

    if (model_name == "mobilenet"):
        base_model = tf.keras.applications.mobilenet.MobileNet(
            weights="imagenet",
            input_shape = (224, 224, 3),
            include_top = True
        )

        cnn_out = base_model.get_layer('global_average_pooling2d').output
        cnn_out = keras.layers.Dropout(0.6)(cnn_out)
        model = tf.keras.models.Model(base_model.input, cnn_out)


    elif model_name == "mobilenetv3":
        base_model = tf.keras.applications.MobileNetV3Large(
            weights="imagenet",
            input_shape=(224,224,3),
            include_top=False
        )

        cnn_out = base_model.get_layer('global_average_pooling2d').output
        cnn_out = keras.layers.Dropout(0.6)(cnn_out)
        model = tf.keras.models.Model(base_model.input, cnn_out)


    elif model_name == "inception":

        base_model = keras.applications.inception_v3.InceptionV3(
            weights='imagenet',
            include_top=True
        )

        model = keras.models.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('avg_pool').output,
            name=model + " without top layer"
        )

    elif model_name == "vgg16":

        base_model = VGG16(
            weights="imagenet",
            input_shape=(224,224,3),
            include_top=False,
            pooling="avg"
        )

        model = keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d_1').output, name="VGG 16 without top layer")
    
    elif model_name == "ResNet50":

        base_model = tf.keras.applications.ResNet15V2(weights="imagenet", input_shape=(224,224,3), include_top=False, pooling="avg")
        print(base_model.summary())
        cnn_out = base_model.get_layer('avg_pool').output
        cnn_out = keras.layers.Dropout(0.6)(cnn_out)

        model = keras.models.Model(inputs=base_model.input, outputs=cnn_out, name="ResNet50WithoutTopLayer")

    elif model_name == "Xception":

        base_model = Xception(weights="imagenet", input_shape=(299,299,3), include_top=True)
        model = keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output, name="ResNet50 without top layer")

    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights="imagenet", input_shape=(224,224,3), include_top=True)

        model = keras.models.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('avg_pool').output
        )

    elif model_name == "lrcn":

        model = Sequential()

        model.add(Conv2D(16, 5, 5, input_shape=(256, 256, 3),  init= "he_normal",  activation='relu',  border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(20, 5, 5, init="he_normal", activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, 3, 3, init= "he_normal",  activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, 3, 3, init= "he_normal",  activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(50, 3, 3, init="he_normal", activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
    
    elif model_name == "noModel":

        model = Sequential()

        model.add(Conv1D(17, 1, input_shape=(25, 17), activation="relu", border_mode="same"))

    else:
        print(model_name)
        raise ValueError("Unknown 2D feature extraction model")

    print(model.summary())

    tu_input_shape = model.input_shape[1:]
    tu_output_shape = model.output_shape[1:]

    print("Expected input shape %s, output shape %s" % (str(tu_input_shape), str(tu_output_shape)))
    print("Supplied input shape %s, output shape %s" % (di_feature["input_shape"], di_feature["output_shape"]))

    if tu_input_shape != di_feature["input_shape"]:
        raise ValueError("Unexpected input shape")
    
    if tu_output_shape != di_feature["output_shape"]:
        raise ValueError("Unexpected output shape")

    return model

def predict_features(frame_base_dir:str, features_base_dir:str, model:keras.Model,
    frames_norm:int=40, output_shape=None, model_name='Other'):
    """
    Used by the MobileNet-LSTM NN architecture.
    The (video) frames (2-dimensional) in frame_base_dir are fed into model (eg MobileNet without top layers)
    and the resulting features are saved to feature_base_dir
    """
    # prepare frame generator without shuffling

    _, h, w, c = model.input_shape
    gen_frames = FramesGenerator(frame_base_dir, 1, frames_norm, h, w, c, 
        classes = None, shuffle=False)

    print("Predict features with %s ... " % model.name)
    count = 0
    # Predict - loop through all samples
    for _, video in gen_frames._videos.iterrows():
        
        # ... frames_base_dir / class / videoname=frame-directory
        original_video = video.frame_dir
        video_name = os.path.normpath(video.frame_dir).split("\\")[-1]
        # video_name = video.frame_dir.split("/")[-1]
        print("Video name")
        print(video_name)
        label = video.label
        feature_path = features_base_dir + "/" + str(label) + "/" + video_name + ".npy"

        # check if already predicted
        if os.path.exists(feature_path):
            print("Video %5d: features already extracted to %s" % (count, feature_path))
            count += 1
            continue
        print('-----------')
        print(original_video)
        print(video_name)
        print('-----------')
        # get normalized frames and predict feature
        x, _ = gen_frames.data_generation(video)
        if  model_name == "noModel":
            feature = x
        else:
            feature = model.predict(x, verbose=0)

        # save to file
        os.makedirs(features_base_dir + "/" + str(label), exist_ok = True)
        np.save(feature_path, feature)

        print("Video %5d: features %s saved to %s" % (count, str(feature.shape), feature_path))
        count += 1

    print("%d features saved to files in %s" % (count+1, features_base_dir))
    return

def features_2D_predict_generator_withResize(frame_base_dir:str, feature_base_dir:str, model:keras.Model, 
    frames_norm:int = 40):
    """
    Used by the MobileNet-LSTM NN architecture.
    The (video) frames (2-dimensional) in frames_base_dir are fed into model (eg MobileNet without top layers)
    and the resulting features are save to feature_base_dir.
    """

    # do not (partially) overwrite existing feature directory
    #if os.path.exists(feature_base_dir): 
    #    warnings.warn("\nFeature folder " + feature_base_dir + " alredy exists, calculation stopped") 
    #    return
	
	# Hamzah: Resize the images
	
    # prepare frame generator - without shuffling!
    _, h, w, c = model.input_shape
    gen_frames = FramesGenerator(frame_base_dir, 1, frames_norm, h, w, c, 
        classes_full = None, shuffle=False)

    print("Predict features with %s ... " % model.name)
    count = 0
    # Predict - loop through all samples
    for _, video in gen_frames.videos_df.iterrows():
        
        # ... frames_base_dir / class / videoname=frame-directory
        video_name = video.frame_dir.split("/")[-1]
        label = video.label
        feature_path = feature_base_dir + "/" + label + "/" + video_name + ".npy"

        # check if already predicted
        if os.path.exists(feature_path):
            print("Video %5d: features already extracted to %s" % (count, feature_path))
            count += 1
            continue

        # get normalized frames and predict feature
        x, _ = gen_frames.data_generation(video)
        feature = model.predict(x, verbose=0)

        # save to file
        os.makedirs(feature_base_dir + "/" + label, exist_ok = True)
        np.save(feature_path, feature_path)

        print("Video %5d: features %s saved to %s" % (count, str(feature.shape), feature_path))
        count += 1

    print("%d features saved to files in %s" % (count+1, feature_base_dir))
    return
	
	
def features_3D_predict_generator(frames_base_dir:str, feature_base_dir:str, 
    model:keras.Model, batch_size:int = 16):
    """
    Used by I3D-top-only model.
    The videos (frames) are fed into model (=I3D without top layers) and
    resulting features are saved to disc. 
    (Later these features are used to train a small model containing 
    only the adjusted I3D top layers.)
    """

    # do not (partially) overwrite existing feature directory
    #if os.path.exists(feature_base_dir): 
    #    warnings.warn("\nFeature folder " + feature_base_dir + " alredy exists, calculation stopped") 
    #    return

    # prepare frame generator - without shuffling!
    _, frames_model, h, w, c = model.input_shape
    gen_frames = FramesGenerator(frames_base_dir, batch_size, frames_model, h, w, c, 
        classes_full = None, shuffle=False)

    # Predict
    print("Predict features with %s ... " % model.name)

    count = 0
    # loop through all samples
    for _, video in gen_frames.videos_df.iterrows():

        # ... frames_base_dir / class / videoname=frame-directory
        video_name = video.sFrameDir.split("/")[-1]
        label = video.label
        feature_path = feature_base_dir + "/" + label + "/" + video_name + ".npy"

        # check if already predicted
        if os.path.exists(feature_path):
            print("Video %5d: features already extracted to %s" % (count, feature_path))
            count += 1
            continue

        # get normalized frames
        frames, _ = gen_frames.data_generation(video)

        # predict single sample
        feature = model.predict(np.expand_dims(frames, axis=0))[0]

        # save to file
        os.makedirs(feature_base_dir + "/" + label, exist_ok = True)
        np.save(feature_path, feature)

        print("Video %5d: features %s saved to %s" % (count, str(feature.shape), feature_path))
        count += 1

    print("%d features saved to files in %s" % (count+1, feature_base_dir))
    
    return    
