"""
https://github.com/FrederikSchorr/sign-language

Define or load a Keras LSTM model.
"""

from multiprocessing.dummy import active_children
import os
import glob
import time
import sys
import warnings

import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from  keras.layers import Bidirectional, LSTM, TimeDistributed, BatchNormalization, Conv3D, MaxPooling1D
from keras.applications.mobilenet import preprocess_input
from keras.optimizers import Adam as adam
from keras.regularizers import l2

from keras.applications.mobilenet import preprocess_input
from keras.layers import ELU, ReLU, LeakyReLU

###########import keras
from keras.layers import Dense, Activation, Flatten, Dropout, ZeroPadding3D, Input, concatenate, Conv1D
from keras.models import Model
from keras import layers

# from transformer_model import TransformerEncoder, PositionalEmbedding

def lstm_build(frames_norm:int, feature_length:int, num_classes:int, dropout:float = 0.5, model_name:str = 'None') : #-> tensorflow.keras.Model

    # Build new LSTM model
    print("Build and compile LSTM model ...")
    
    print('Model Name: ', model_name)			

    if model_name == 'lstm':
        model = tf.keras.models.Sequential()
        model.add(keras.layers.LSTM(feature_length * 1, return_sequences=False, dropout=dropout))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
    

    else:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.GRU(2024, return_sequences=True,
            input_shape=(frames_norm, feature_length),
            dropout=0.7))
        model.add(tf.keras.layers.GRU(2024, return_sequences=False, dropout=0.7))
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.6))
        model.add(tf.keras.layers.Dropout(0.6))

        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    print("================MODEL SUMMARY==================")
    model.summary()
    print("================END OF MODEL SUMMARY==================")

    return model

def lstm_build_multi(frames_norm:int, feature_length_01:int, feature_length_02:int, num_classes:int, dropout:float = 0.5, model_name:str = 'None'): #-> keras.Model:

    # Build new LSTM model
    print("Build and compile the model ...")

    
    input_01 = Input(shape=(frames_norm, feature_length_01))         
    model_01_01 = tf.keras.layers.LSTM(feature_length_01 * 1, return_sequences=True, input_shape=(frames_norm, feature_length_01), dropout=dropout)(input_01)    
    model_01_02 = tf.keras.layers.LSTM(feature_length_01 * 1, return_sequences=False, dropout=dropout)(model_01_01)


    input_02 = Input(shape=(frames_norm, feature_length_02))         
    model_02_01 = tf.keras.layers.LSTM(1024, return_sequences=True, input_shape=(frames_norm, feature_length_02), dropout=0.5)(input_02)    
    model_02_02 = tf.keras.layers.LSTM(1024, return_sequences=False, dropout=0.5)(model_02_01)

    merged_layers = concatenate([model_01_02, model_02_02])         
    fc = Dense(num_classes, activation='softmax')(merged_layers)
    
    model = Model([input_01, input_02], [fc]) 
       
    
    model.summary()

    return model

def lstm_build_multi_single(frames_norm:int, feature_length_01:int, feature_length_02:int, num_classes:int, dropout:float = 0.5, model_name:str = 'None'): #-> keras.Model:

    # Build a fused LSTM and CNN (LSTM for frames and CNN for a single image per video)
 

    ## Model 01
    input_frames =  Input(shape=(frames_norm, feature_length_01), name='input_frames')
    
    x1 = LSTM(2048, return_sequences=True, input_shape=(frames_norm, feature_length_01), dropout=0.7)(input_frames)    
    x1 = LSTM(2048, return_sequences=True, dropout=0.7)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.6)(x1)
    x1 = tf.keras.layers.Dropout(0.6)(x1)

    #########x1 = BatchNormalization( axis = -1 )(x1)


    ## Model 02
    img_size = 224
    input_img = tf.keras.layers.Input(shape=(img_size, img_size, 3), name='input_img')
    x2 = preprocess_input(input_img)
    x2 = tf.keras.applications.MobileNet(weights="imagenet",include_top=False, input_tensor=x2)
    x2 = keras.layers.GlobalAveragePooling2D()(x2.output)
    x2 = keras.layers.Dropout(0.6)(x2)
    x2 = keras.layers.Dropout(0.6)(x2)

    #########x2 = BatchNormalization(axis = -1 )(x2)

    ##
    x = concatenate([x1, x2])
    x = BatchNormalization(axis = -1 )(x)
    x = x[:,:,np.newaxis]
    x = tf.keras.layers.Conv1D(256, 7, activation='relu')(x)
    # x = keras.layers.Dropout(0.6)(x)
    # x = keras.layers.Dropout(0.6)(x)

    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dropout(0.7)(x)
    x = keras.layers.Dropout(0.6)(x)
    x = Flatten()(x)

    fc = layers.Dense(num_classes,  activation = "softmax")(x)
    #model = tf.keras.models.Model([input_frames, model_cnn.input], fc)
    model = tf.keras.models.Model(inputs=[input_frames, input_img], outputs=fc)

    
    model.summary()

    return model


def pretrainedModel(img_size, model_name, num_classes, retrain_model=False):

    input_img = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    #normalization_layer = tf.keras.layers.Rescaling(1./255)
    if model_name == 'mobileNet':
        input_img = preprocess_input(input_img)
        model_cnn = tf.keras.applications.MobileNet(weights="imagenet",include_top=False, input_tensor=input_img)
    elif model_name == 'InceptionV3':
        model_cnn = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False, input_shape=(256, 256, 3))
    if retrain_model:
        for layer in model_cnn.layers[:-4]:
            layer.trainable = True

    cnn_out = keras.layers.GlobalAveragePooling2D()(model_cnn.output)
    cnn_out = keras.layers.Dropout(0.6)(cnn_out)
    cnn_out = keras.layers.Dropout(0.6)(cnn_out)

    #cnn_out = keras.layers.Dense(num_classes, activation="softmax")(cnn_out)
    model = tf.keras.models.Model(model_cnn.input, cnn_out)
    #model.compile(metrics=['accuracy'], loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=adam(learning_rate=1e-4))
    
    return model


def lstm_load(path:str, frames_norm:int, feature_length:int, num_classes:int) -> tf.keras.Model:

    print("Load trained LSTM model from %s ..." % path)
    model = tf.keras.models.load_model(path)
    
    input_shape = model.input_shape[1:]
    output_shape = model.output_shape[1:]
    print("Loaded input shape %s, output shape %s" % (str(input_shape), str(output_shape)))

    if input_shape != (frames_norm, feature_length):
        raise ValueError("Unexpected LSTM input shape")
    if output_shape != (num_classes, ):
        raise ValueError("Unexpected LSTM output shape")

    return model