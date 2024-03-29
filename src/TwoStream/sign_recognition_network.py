from dataclasses import asdict

import os
import time
import seaborn as sn
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, CSVLogger # TimeHistory
from keras.optimizers import Adam

from math import ceil

import keras

import tensorflow as tf

from data_generator import VideoClasses, FeaturesGeneratorMultiInput
from lstm_model import lstm_build_multi_single

def read_data(data_path, convert_to_int=False, ext=".npy"):
    """
    read csvg data and return sample paths and labels

    :param data_path: path to csv file with relevant information
    :param convert_to_int: if true, convert all laabels to integers
    :param ext: extension to add to sample path for example .npy
    """
    data_all = pd.read_csv(data_path, names=["index", "cat", "path", "frame_num", "signer_id"], header=None)

    if convert_to_int:
        samples_df = data_all.path.to_frame()
        samples_df.path = samples_df.path + ext
        labels = pd.get_dummies(data_all.path.apply(lambda s: s.split("/")[-2]).to_numpy().astype(int)).to_numpy()
    else:
        samples_df = data_all["path"].copy().to_frame()
        samples_df = samples_df.path + ext
        labels = data_all.path.apply(lambda s: s.split("/")[-2])
    
    return samples_df, labels

def train_feature_generator(model_dir:str, log_path:str, model:keras.Model, classes:VideoClasses, batch_size:int=16, 
                            epochs:int=100, learning_rate:float=1e-4, csv_file=False, train_path_one=None, 
                            train_path_two=None, load_model=False):
    """
    train model

    :param model_dir: directory to save model
    :param log_path: directory to log model performance
    :param model: model to be trained
    :param classes: video classes object
    :param batch_size: batch size hyperparameter for training
    :param epochs: number of epochs to train for
    :param learning_rate: learning rate for training
    :param csv_file: if true, load data using csv file
    :param train_path_one: training path
    :param train_path_two: validation path
    :param load_model: if true, load saved model
    """
    if csv_file:
        samples_df, labels = read_data(train_path_one)

        indices = np.arange(samples_df.shape[0])
        train_data_one, val_data_one, _, _ = train_test_split(samples_df, labels, indices, test_size=0.2, stratify=labels)
        val_labels_one =  pd.get_dummies(val_data_one.path.apply(lambda s: s.split("/")[-2]).to_numpy().astype(int)).to_numpy() # hope this good
        train_data_one.reset_index(drop=True, inplace=True)
        val_data_one.reset_index(drop=True, inplace=True)
        print(train_data_one.shape)

    if load_model == False:
        gen_training_features = FeaturesGeneratorMultiInput(train_data_one, train_path_two, batch_size, model.input_shape[0][1:], model.input_shape[1][1:], classes._classes, shuffle=True)
        gen_validation_features = FeaturesGeneratorMultiInput(val_data_one, train_path_one, batch_size, model.input_shape[0][1:], model.input_shape[1][1:], classes._classes, shuffle=True)

    csv_logger = CSVLogger(log_path.split(".")[0] + "-acc.csv")

    early_stopper = EarlyStopping(monitor="val_loss", patience=10)

    os.makedirs(model_dir, exist_ok=True)
    best_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath = model_dir + "/model-best.h5",
        verbose = 1, save_best_only = True)
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print("Fit with generator, learning rate %f ..." % learning_rate)
    n_points = val_labels_one.shape[0]
    steps_per_epoch = ceil(n_points / batch_size)

    if load_model:
        model = []
        model = tf.keras.models.load_model(saved_model)
        es = 0
    else:
        hist = model.fit(
            gen_training_features,
            validation_data=gen_validation_features,
            epochs=epochs,
            workers=1,
            use_multiprocessing=False,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            callbacks=[csv_logger, best_checkpoint, early_stopper])
        
        gen_training_features = []
        gen_validation_features = []

        es = early_stopper.stopped_epoch
    
def print_model_times(time_info):
    """
    prints historical model times

    :param time_info: time info object
    """
    # print and save model times
    print("Total time:")
    print(np.sum(time_info))

def write_CSV_file(data, filename, path_name):
    """
    write data to csv file

    :param data: data to be saved to csv
    :param filename: filename to save csv under
    :param path_name: directory path to save file
    """
    pd.DataFrame(data).to_csv(os.path.join(path_name, filename), sep=",")

def write_results(hist, exp_path):
    """
    write model history to file

    :param hist: model history object
    :param exp_path: directory to save results
    """
    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    train_acc = hist.history["accuracy"]
    val_acc = hist.history["val_accuracy"]

    data_frame_data = np.transpose([train_loss, train_acc, val_loss, val_acc])

    write_CSV_file(data_frame_data, "train_val_losses_acc.csv", exp_path)

def test(rm, es, main_exp_folder, load_to_memory, class_limit, test_generator=None, X_test=None, y_test=None):
    """
    test model

    :param rm: loaded model
    :param es: used to pass early stopping epoch for writing?
    :param main_exp_folder: folder to save reuslts
    :param load_to_memory: if true, use supplied data and labels, instead of test_generator
    :param test_generator: generator for test data
    :param X_test: testing data
    :param y_test: testing labels
    """
    if load_to_memory:
        loss, acc = rm.evaluate(X_test, y_test, verbose=0)
    else:
        loss, acc = rm.evaluate(test_generator, verbose=0)

    print("\nTesting loss: {}, acc: {}\n".format(loss, acc))

    f = open(main_exp_folder + "/testAccuracy.txt", "a")

    early_stopping_epoch = es
    f.write("Early stopped at:\t" + str(early_stopping_epoch) + "\t" + str(loss) + " " + str(acc) + "\n")
    
    if load_to_memory:
        Y_pred = rm.predict(X_test)
        y_pred = np.argmax(Y_pred, axis=1)
        print_confusion_matrix(X_test, y_test, class_limit, y_pred, main_exp_folder)
    
    else:
        Y_pred = rm.predict_generator(test_generator,
                                      workers = 1,
                                      use_multiprocessing = False,
                                      verbose = 1)
        
        y_pred = Y_pred.argmax(axis=1)

        print_confusion_matrix(y_test, class_limit, y_pred, main_exp_folder)

def print_confusion_matrix(y_test, n_classes, y_pred, main_exp_folder):
    """
    prints and saves a confusion matrix

    :param y_test: actual class labels
    :param n_classes: number of classes
    :param y_pred: predicted class labels
    :param main_exp_folder: main directory to save testing results
    """
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred)
    print("Confusion Matrix: ", confusion_matrix.shape)
    filename = os.path.join(main_exp_folder, "CM.csv")

    pd.DataFrame(cm).to_csv(filename, sep=",")
    pd.DataFrame(y_test.argmax(axis=1)).to_csv(os.path.join(main_exp_folder, "testLabels.csv"), sep=",")
    pd.DataFrame(y_pred).to_csv(os.path.join(main_exp_folder, "predicted_testLabels.csv"), sep=",")

    # Visualising of confusion matrix
    plot_cm = False
    if plot_cm:
        df_cm = pd.DataFrame(cm, range(n_classes), range(n_classes))
        plt.figure(figsize=(n_classes, n_classes))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})
        plt.savefig(os.path.join(main_exp_folder, "cm.png"))

    class_metrics = classification_report(y_test.argmax(axis=1), y_pred, output_dict=True)
    print(classification_report(y_test.argmax(axis=1), y_pred))
    df = pd.DataFrame(class_metrics).transpose()
    pd.DataFrame(df).to_csv(os.path.join(main_exp_folder, "ResultsMetrics.csv"), sep=",")

def visualise_hist(hist, exper_name, use_batch):
    """
    print and save training history

    :param hist: model history object
    :param exper_name: name to save experiment under
    :param use_batch: if true, show plots
    """
    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    train_acc = hist.history["accuracy"]
    val_acc = hist.history["val_accuracy"]

    xc = range(len(val_acc))

    plt.figure(3)
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Model loss")
    plt.grid(True)
    plt.legend(["Train", "Validation"])
    plt.savefig(exper_name + "/loss.png")
    if use_batch == 1:
        plt.show()

    plt.figure(4)
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.grid(True)
    plt.legend(["Train", "Validation"], loc=4)

    plt.savefig(exper_name + "/acc.png")
    if use_batch == 1:
        plt.show()

def train_mobile_lstm(video_set, feature_one, feature_two, exp_full_path=None, val_available=None, 
                      csv_file=None, train_path_one=None, val_path_one=None, train_path_two=None, val_path_two=None,
                        class_file=None, image_feature_dir=None, load_model=False, saved_model=None):
    
    """
    trains mobile lstm

    :param model_dir: directory to save model
    :param log_path: directory to log model performance
    :param model: model to be trained
    :param classes: video classes object
    :param batch_size: batch size hyperparameter for training
    :param epochs: number of epochs to train for
    :param learning_rate: learning rate for training
    :param csv_file: if true, load data using csv file
    :param train_path_one: training path
    :param train_path_two: validation path
    :param load_model: if true, load saved model
    """
    
    if os.path.exists(exp_full_path) == False:
        os.mkdir(exp_full_path, 0o755)

    model_dir = exp_full_path

    print("\nStarting training...")
    print(os.getcwd())

    # read the classes
    classes = VideoClasses(class_file)

    log_path = os.path.join(exp_full_path, time.strftime("%Y%m%d-%H%M", time.gmtime()) + "-%s%03d-image-mobile-lstm.csv"%(video_set["name"], video_set["classes_count"]))
    print("Image log: %s" % log_path)

    model_image = lstm_build_multi_single(video_set["frames_norm"], feature_one["output_shape"][0], feature_two["output_shape"][0], classes.classes_count, dropout=0.5)

    train_feature_generator(image_feature_dir, model_dir, log_path, model_image, classes,
                            batch_size=32, epochs=100, learning_rate=1e-4, exp_full_path=exp_full_path, 
                            val_available=val_available, csv_file=csv_file, train_path_one=train_path_one,
                            val_path_one=val_path_one, train_path_two=train_path_two, val_path_two=val_path_two,
                            load_model=load_model, saved_model=saved_model)
    
    return

if __name__ == '__main__':
    # Implement later :)
    video_set = {
        "name" : "multi-sign",
        "classes" : 29,
        "frames_norm" : 18,
        "min_dim" : 224,
        "shape" : 30,
        "frames_avg" : 50,
        "avg_duration" : 2.0
    }

    feature_one = {
        "name" : "mobilenet",
        "input_shape" : (224, 224, 3),
        "output_shape" : (1024, )
    }

    feature_two = {
        "name" : "mobilenet",
        "input_shape" : (224, 224, 3),
        "output_shape" : (1024, )
    }

    
    class_file_all = "./DataInfo/ELAR_classes.csv"

    image_feature_dir_all = "" # Is this required

    dataset_home_path = "./Data/ELAR"

    train_path_one = "./DataInfo/ELAR.csv"

    val_path_one = ""

    test_path_one = "" # I don't want a test path for now

    # Possibly need to update things in these paths
    data_set_home_path_forward = "./Data/ELAR/fusion/backward"
    data_set_home_path_backward = "./Data/ELAR/fusion/backward"
    data_set_home_path_both = "./Data/ELAR/fusion/both"
    data_set_home_path_star = "./Data/ELAR/star"

    train_path_two = [
        data_set_home_path_forward,
        data_set_home_path_backward,
        data_set_home_path_both,
        data_set_home_path_star,
    ]

    val_path_two = ""

    experiment_paths = "" # What this?

    load_model = [False] * 3
    saved_model = [None] * 3

    csv_file = [True] * 3
    val_available = [False] * 3

    i = 0
    print("==============================")
    print(len(experiment_paths))
    while i < len(experiment_paths):
        experiment_path = experiment_paths[i]
        print(experiment_path)
        val_available = val_available[i]
        full_experiment_path = os.path.join(os.getcwd(), "results", experiment_path)
        train_mobile_lstm(video_set, feature_one, feature_two, exp_full_path=full_experiment_path,
                          exp_path=experiment_path, val_available=val_available, csv_file=csv_file,
                          train_path_one=train_path_one, val_path_one = val_path_one, 
                          train_path_two=train_path_two[i], val_path_two=val_path_two[i],
                          class_file=class_file_all, image_feature_dir=image_feature_dir_all[i],
                          load_model=load_model[i], saved_model=saved_model[i])
        i += 1
