"""
Accept two data sources and fuse them at the classification layer
@author: hluqman@kfupm.edu.sa
"""

import numpy as np
import tensorflow as tf
import seaborn as sn
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from keras.optimizers import Adam as adam
from keras.callbacks import EarlyStopping, CSVLogger
from keras.models import Sequential
from keras import layers
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.utils import shuffle 


from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, History

def write_results(hist, exp_path):
    """
    Writes the results of the given history object
    to a csv in the provided path

    :param exp_path: path to save results
    """
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['accuracy']
    val_acc=hist.history['val_accuracy']
    data_frame = np.transpose([train_loss, train_acc, val_loss, val_acc])
    write_csv_file(data_frame , 'train_val_losses_acc.csv', exp_path)


def save_model_times(time_info, exp_path):
    """
    Print and save time info to provided path

    :param time_info: time info data
    :param exp_path: path to save model times
    """
    print('Total time:')
    print(np.sum(time_info))
    write_csv_file(time_info , 'model_times.csv', exp_path)

def write_csv_file(data, file_name, path_name):
    """
    save data to csv

    :param data: csv data frame to be saved
    :param file_name: file name for new csv
    :param path_name: path to save file
    """
    pd.DataFrame(data).to_csv(os.path.join(path_name, file_name), sep=',')

def visualise_hist(hist, experiment_name, use_batch=0):
    """
    Plots the model training loss, val_loss, accuracy, and val accuracy history

    :param hist: model history object
    :param experiment_name: experiment name to save plot under
    :param use_batch: if 1, show plot
    """
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['accuracy']
    val_acc=hist.history['val_accuracy']
    
    xc = range(len(val_acc))

    plt.figure(3)
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model loss')
    plt.grid(True)
    plt.legend(['Train', 'Validation'])
    plt.savefig(experiment_name + '/loss.png')
    if use_batch == 1:
        plt.show()

    plt.figure(4)
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy')
    plt.grid(True)
    plt.legend(['Train', 'Validation'], loc=4)

    plt.savefig(experiment_name + '/acc.png')
    if use_batch == 1:
        plt.show()

def print_confusion_matrix(y_test, n_classes, y_pred, main_exp_folder):
    """
    Print the confusion matrix given a list of actual and predicted values
    Save confusion matrix to experiment folder

    :param y_test: validation/test labels
    :param n_classes: number of classes
    :param y_pred: predicted labels
    :param main_exp_folder: experiment folder to save confusion matrix
    """
    cf_matrix = confusion_matrix(y_test, y_pred)
    filename = os.path.join(main_exp_folder,'CM.csv')

     
    pd.DataFrame(cf_matrix).to_csv(filename, sep=',')
    pd.DataFrame(y_test).to_csv(os.path.join(main_exp_folder ,'TestLabels.csv'), sep=',')
    pd.DataFrame(y_pred).to_csv(os.path.join(main_exp_folder ,'predicted_testLabels.csv'), sep=',')
       
    # Visualizing of confusion matrix
    plotCM = True
    if plotCM:
        classes = ('Arrive', 'Bed', 'Bird', 'Boy', 'Come', 'Day', 'Deer', 'Frog', 'Girl', 'Good', 'Lady', 'Laugh', 
               'Man', 'Night', 'People', 'Rabbit', 'Real', 'Same', 'Say', 'Sheep', 'Slow', 'Sprint', 'Think', 
               'Tortoise', 'What', 'Where', 'Window', 'Wolf', 'Yell')
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                             columns = [i for i in classes])
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join(main_exp_folder,'cm.png'))
    
    classes = ['Arrive', 'Bed', 'Bird', 'Boy', 'Come', 'Day', 'Deer', 'Frog', 'Girl', 'Good', 'Lady', 'Laugh', 
               'Man', 'Night', 'People', 'Rabbit', 'Real', 'Same', 'Say', 'Sheep', 'Slow', 'Sprint', 'Think', 
               'Tortoise', 'What', 'Where', 'Window', 'Wolf', 'Yell']
    class_metrics = classification_report(y_test, y_pred, target_names=classes)
    with open(os.path.join(main_exp_folder, 'ResultMetrics.txt'), "w") as text_file:
        text_file.write(class_metrics)


def pretrained_model(img_size, model_name, retrain=False):
    """
    return a compiled feature extraction model

    :param img_size: width and height of input image
    :param model_name: cnn model name to use
    :param retrain: make layers of the feature extractor trainable
    :return: pretrained feature extractor model
    """
    input_img = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    print("Input image")
    print(input_img)
    if model_name == 'mobileNet':
        input_img = keras.applications.mobilenet.preprocess_input(input_img)
        model_cnn = tf.keras.applications.MobileNet(weights="imagenet", include_top=False, input_tensor=input_img)
    elif model_name == 'InceptionV3':
        model_cnn = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    elif model_name == 'ResNet':
        input_img = keras.applications.resnet50.preprocess_input(input_img)
        model_cnn = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=input_img)
    if retrain:
        for layer in model_cnn.layers[:-4]:
            layer.trainable = True

    cnn_out = keras.layers.GlobalAveragePooling2D()(model_cnn.output)
    cnn_out = keras.layers.Dropout(0.6)(cnn_out)
    cnn_out = keras.layers.Dropout(0.6)(cnn_out)

    cnn_out = keras.layers.Dense(class_limit, activation="softmax")(cnn_out)
    model = tf.keras.models.Model(model_cnn.input, cnn_out)
    if dataFormat =='csv':
        model.compile(metrics=['accuracy'], loss="categorical_crossentropy", optimizer=adam(learning_rate=1e-4, ema_momentum=0.9))
    else:
        model.compile(metrics=['accuracy'], loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=adam(learning_rate=1e-4))

    return model

def append_image_extention(img_path):
    """
    appends jpg extension to image path

    :param img_path: path to image
    """
    return img_path + '.jpg'

def main(model_name, retrain, train_path, test_path, main_exp_folder, model_dest, 
         class_limit, img_size, data_format='folder', load_model=False, saved_model=None):
    """
    trains accumulative motion network

    :param model_name: starting convolutional model
    :param retrain: if true, train the last four layers of the chosen cnn
    :param train_path: path to training data
    :param test_path: path to testing data
    :param main_exp_folder: path to save experiment results
    :param model_dest: destination to save trained model
    :param class_limit: number of classes
    :param img_size: input image size in pixels
    :param data_format: collect data based on folder structure or csv file
    :param load_model: if true, load pretrained model
    :param saved_model: saved model to load
    """
    if os.path.exists(main_exp_folder)  == False :
        os.mkdir(main_exp_folder, 0o755)

    print('---------------------------------------START----------------------------')
    print(main_exp_folder)
    print('------------------------------------------------------------------------')
    datagen_train = ImageDataGenerator(rotation_range=10, zoom_range=0.2, horizontal_flip=True)
    datagen_val = ImageDataGenerator(seed)

    if data_format == 'csv':
        # Generate testing data from csv
        train_df = pd.read_csv(train_path, names=['index','label','path','framesN','signerID'], header=None, dtype=str)
        train_df = shuffle(train_df)
        train_df["path"] = train_df["path"].apply(append_image_extention)

        if test_path != None:
            test_df = pd.read_csv(test_path, names=['index','label','path','framesN','signerID'], header=None, dtype=str)
            test_df["path"] = test_df.path.apply(append_image_extention)
            print(test_df)
            train_ds = datagen_train.flow_from_dataframe(dataframe=train_df, directory=None, x_col="path", y_col="label", 
                                                         batch_size=8, class_mode='categorical', subset = "training", 
                                                         target_size=(img_size, img_size), shuffle=True, seed=42)
            
            val_ds  = datagen_train.flow_from_dataframe(dataframe=train_df,  directory=None, x_col="path", y_col="label", 
                                                        batch_size=8, class_mode='categorical', subset = "validation", 
                                                        target_size=(img_size, img_size), shuffle=True, seed=42)

        else:
            y_ = train_df['label'].to_numpy()
            data_pdf = train_df[['path', 'label']]
            X_train_df, test_df = train_test_split(data_pdf, test_size=0.20, stratify=y_)

            print(data_pdf)
            print(X_train_df)

            datagen_train = ImageDataGenerator(validation_split=0.10, rotation_range=10, zoom_range=0.2, horizontal_flip=True)

            train_ds = datagen_train.flow_from_dataframe(dataframe=X_train_df, directory=None, x_col="path", y_col="label", 
                                                         batch_size=8, class_mode='categorical', subset="training", 
                                                         target_size=(img_size, img_size), shuffle=True, seed=42)
            
            val_ds  = datagen_train.flow_from_dataframe(dataframe=X_train_df,  directory=None, x_col="path", y_col="label", 
                                                        batch_size=8, class_mode='categorical', subset = "validation", 
                                                        target_size=(img_size, img_size), shuffle=True, seed=42)
        print('///////////////////////')
        print(train_ds.n)
         
        STEP_SIZE_TRAIN = train_ds.n // 8
        STEP_SIZE_VALID = val_ds.n // 8

    else:
        """
        Generate data from a training and validation directory
        """
        train_ds = datagen_train.flow_from_directory(train_path,  batch_size=8, class_mode='sparse', 
                                                     target_size=(img_size, img_size), shuffle=True, 
                                                     seed=42, subset = "training")
        
        val_ds = datagen_train.flow_from_directory(train_path,  batch_size=8, class_mode='sparse', 
                                                   target_size=(img_size, img_size), shuffle=False, 
                                                   seed=42, subset = "validation")
        
        STEP_SIZE_TRAIN = train_ds.samples // 8
        STEP_SIZE_VALID = val_ds.samples // 8

    # model 
    if (model_name =='mobileNet' or model_name == 'ResNet') and load_model == False:
        model = pretrained_model(img_size, model_name, retrain)

    early_stopper = EarlyStopping(monitor='val_loss', patience=10)
    time_callback = History()
    best_checkpoint = keras.callbacks.ModelCheckpoint(filepath = main_exp_folder + "/model-best.h5",  
                                                      verbose=1, save_best_only=True)

    if load_model:
        model = keras.models.load_model(saved_model)
        print(model.summary())
        es = 0
    else:    
        hist = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[time_callback, best_checkpoint, early_stopper])
        model.save(model_dest) 
        model = keras.models.load_model(filepath = main_exp_folder + "/model-best.h5")

        write_results(hist, main_exp_folder)     
        visualise_hist(hist, main_exp_folder)

        score, acc = model.evaluate(val_ds)
        print('val score:', score)
        print('val accuracy:', acc)

    model.load_weights(main_exp_folder + "/model-best.h5")
    Y_pred = model.predict(val_ds)
    y_pred = np.argmax(Y_pred, axis=1)
    pd.DataFrame(y_pred).to_csv(os.path.join(main_exp_folder, 'predicted_testLabels.csv'), sep=',')

    ### Get labels
    y_test = val_ds.labels 
    pd.DataFrame(y_test).to_csv(os.path.join(main_exp_folder, 'testLabels.csv'), sep=',')

    print_confusion_matrix(y_test, class_limit, y_pred, main_exp_folder)

"""Configure our models"""
model_name = 'ResNet'
input_size = 224
retrainModel = False
dataFormat = 'csv' # 'csv' for csv files or 'folder' to read from folders

"""Configure the rest of our settings"""
train_path = "../../data/elar/star/train" #contains train signs after forward fusion
main_exp_folder = './results/elar/star/resnet3'
model_dest = os.path.join(main_exp_folder, 'Last.h5')
class_limit = 29
load_model = False
savedModel = './results/elar/star/resnet3/model-best.h5'

main(model_name, retrainModel, train_path, None, main_exp_folder, model_dest, class_limit,
      input_size, dataFormat, load_model, savedModel)