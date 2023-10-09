"""
	https://github.com/FrederikSchorr/sign-language

Train a LSTM neural network to classify videos. 
Requires as input pre-computed features, 
calculated for each video (frames) with MobileNet.
"""

import os
import glob
import time
import seaborn as sn
from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, History

import keras

from datagenerator import VideoClasses, FeaturesGenerator, FeaturesGenerator_withSplitting
from lstm_model import lstm_build


def train_feature_generator(feature_dir:str, model_dir:str, log_path:str, model:keras.Model, classes_object: VideoClasses,
    batch_size:int=16, num_epochs:int=1, learning_rate:float=1e-4, exp_full_path=None, val_available=True, csv_file=False, 
    train_path=None, val_path=None, test_path=None, video_set=None, feature=None, load_model=False, saved_model=None):
    """
    The training code given a model and appropriate parameters

    :param feature_dir: path to feature directory
    :param model_dir: path to the previously trained model directory
    :param log_path: path to log directory
    :param model: sequence model being trained
    :param classes_object: video classes object
    :param batch_size: batch size hyperparameter
    :param num_epochs: maximum number of epochs to train model for
    :param learning_rate: learning rate of sequence model
    :param exp_full_path: path to save model training information
    :param val_available: if true, separate validation set can be used
    :param csv_file: if true, csv file used to construct dataloader
    :param train_path: path to training data
    :param val_path: path to validation data
    :param test_path: path to testing data
    :param video_set: describes the training set used - usually ELAR
    :param feature: describes the output from the feature extractor
    :param load_model: if true, load pretrained sequence model
    :param saved_model: path to saved model 
    """
    if csv_file:
        print('===============================================')
        train_df = pd.read_csv(train_path)
        train_data = train_df['path'].copy().to_frame()
        train_data.path = train_data.path + ".npy"

        labels = train_data.path.apply(lambda s: s.split("/")[-2])

        if test_path == None:
            train_data, test_samples_df = train_data, labels 
            train_data.reset_index(drop=True, inplace=True)
            labels = train_data.path.apply(lambda s: s.split("/")[-2])
        else:
            test_data = pd.read_csv(test_path,names=['index','cat','path','frame_count','signerID'],header=None)
            test_samples_df = test_data.path.to_frame()
            test_samples_df.path = test_samples_df.path + ".npy"
            labels_test =  pd.get_dummies(test_data.path.apply(lambda s: s.split("\\")[-2]).to_numpy().astype(int)).to_numpy()

        # validation part
        train_data, val_data, y_train, y_val = train_test_split(train_data, labels, test_size=0.20, random_state=42, stratify=labels)
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)
        labels_val = pd.get_dummies(val_data.path.apply(lambda s: s.split("/")[-2]).to_numpy().astype(int)).to_numpy()
    else:            
        samples_df = pd.DataFrame(sorted(glob.glob(feature_dir+ "/train" + "/*/*.npy")), columns=["path"])
        labels =  samples_df.path.apply(lambda s: s.split("/")[-2])
        
        train_data, val_data, y_train, y_val = train_test_split(samples_df, labels, test_size=0.30, random_state=42, stratify=labels)
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)

        test_samples_df = pd.DataFrame(sorted(glob.glob(feature_dir+ "/test" + "/*/*.npy")), columns=["path"])
        labels_test =  pd.get_dummies(test_data.path.apply(lambda s: s.split("/")[-2]).to_numpy().astype(int)).to_numpy()

    print(model.input_shape)
    gen_features_train = FeaturesGenerator_withSplitting(train_data, batch_size, model.input_shape[1:], classes_object._classes, True, video_set=video_set, feature=feature)
    gen_features_val = FeaturesGenerator_withSplitting(val_data, batch_size, model.input_shape[1:], classes_object._classes, True, video_set=video_set, feature=feature)
        
    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger(log_path.split(".")[0] + "-acc.csv")

	# Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(monitor='val_loss', patience=20)

    # Helper: Save the model
    os.makedirs(model_dir, exist_ok=True)
    last_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath = model_dir + "/" + (log_path.split("/")[-1]).split(".")[0] + "-last.h5",
        verbose = 0)
    best_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath = model_dir + "/model-best.h5",
        verbose=1, save_best_only=True)

    optimizer = keras.optimizers.Adam(lr = 1e-4) #1e-4
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    time_callback = History()
    
    # Fit!
    print("Fit with generator, learning rate %f ..." % learning_rate)
    print('******************************')
    print(load_model)
    print('******************************')

    if load_model:
        model = tf.keras.models.load_model(saved_model)
        es = 0
    else:    
        hist = model.fit(
            gen_features_train,
            validation_data = gen_features_val,
            epochs = num_epochs,
            workers = 1, #4,                 
            use_multiprocessing = False, #True,
            verbose = 1, 
            callbacks=[csv_logger, time_callback, best_checkpoint, early_stopper]) 
        
        model = tf.keras.models.load_model(model_dir + "/model-best.h5")
        es = early_stopper.stopped_epoch
        write_results(hist, exp_full_path, 0)
        visualize_hist(hist, exp_full_path, 0)

    test_loss, test_acc = model.evaluate(gen_features_val, verbose=0)
    print('\nTesting loss: {}, acc: {}\n'.format(test_loss, test_acc))
    test(model, es, exp_full_path,False, 50, gen_features_val, None, labels_val) 
    return

def save_model_times(time_info, exp_path):
    """
    Writes the model times to a csv file

    :param time_info: model time info object
    :param exp_path: path to save model times
    """
    # Print and save model times
    print('Total time:')
    print(np.sum(time_info))
    write_csv_file(time_info , 'model_times.csv', exp_path)

def write_csv_file(data, file_name, path_name):
    """
    save data to csv file

    :param data: data to be saved
    :param file_name: file name to save data under
    :param path_name: path to save new csv file
    """
    pd.DataFrame(data).to_csv(os.path.join(path_name,file_name), sep=',')

def write_results(hist, exp_path):
    """
    Writes the results to a csv file

    :param hist: model history object
    :param exp_path: path to save model historical accuracy and loss
    """
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['accuracy']
    val_acc=hist.history['val_accuracy']
    dataframe_data = np.transpose([train_loss, train_acc, val_loss, val_acc])
    write_csv_file(dataframe_data , 'train_val_losses_acc.csv', exp_path)


def test(rm, es, main_exp_folder, load_to_memory, class_limit, test_generator=None, X_test=None, y_test=None):
    """
    Test the model with any prepared testing data

    :param rm: model to be tested
    :param es: early stopping epoch
    :param main_exp_folder: folder to store experiment results
    :param load_to_memory: if false, use test generator instead of np array
    :param class_limit: maximum number of classes
    :param test_generator: the test generator used
    :param X_test: test featuress
    :param y_test: test labels
    """
    if load_to_memory:
        #use X_test, y_test
        loss, acc = rm.evaluate(X_test, y_test, verbose=0)
    else:
        loss, acc = rm.evaluate_generator(test_generator, verbose=0)
    
    print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
    
    f = open(main_exp_folder+ '/testAccurracy.txt',"a")
    #f.write(' '.join(expInfo)+'\n')
    early_stopping_epoch = es
    f.write('Early stopped at:\t' +str(early_stopping_epoch) + '\t'+ str(loss) + ' '+ str(acc)+ '\n')
   
    if load_to_memory: 
        Y_pred = rm.predict(X_test)
        y_pred = np.argmax(Y_pred, axis=1)
        print_confusion_matrix(X_test, y_test, class_limit, y_pred, main_exp_folder)

    else:
        Y_pred = rm.predict_generator(test_generator)
        y_pred = np.argmax(Y_pred, axis=1)
        print_confusion_matrix(X_test, y_test, class_limit, y_pred, main_exp_folder)

def print_confusion_matrix(y_test, n_classes, y_pred, main_exp_folder):
    """
    Saves and prints the confusion matrix

    :param y_test: actual test labels
    :param n_classes: number of classes
    :param y_pred: predicted test labels
    :param main_exp_folder: path to save confusion matrix
    """
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred)
    print('Confusion Matrix: ', cm.shape)
    file_name=os.path.join(main_exp_folder,'CM.csv')
     
    pd.DataFrame(cm).to_csv(file_name, sep=',')
    pd.DataFrame(y_test.argmax(axis=1)).to_csv(os.path.join(main_exp_folder ,'testLabels.csv'), sep=',')
    pd.DataFrame(y_pred).to_csv(os.path.join(main_exp_folder ,'predicted_testLabels.csv'), sep=',')
       
    # Visualizing of confusion matrix
    plotCM = False
    if plotCM:
        df_cm = pd.DataFrame(cm, range(n_classes), range(n_classes))
        plt.figure(figsize = (n_classes, n_classes))
        sn.set(font_scale=1.4)#for label size
        sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
        plt.savefig(os.path.join(main_exp_folder,'cm.png'))
    
    class_metrics = classification_report(y_test.argmax(axis=1), y_pred, output_dict=True)
    print(classification_report(y_test.argmax(axis=1), y_pred))
    df = pd.DataFrame(class_metrics).transpose()
    pd.DataFrame(df).to_csv(os.path.join(main_exp_folder, 'ResultMetrics.csv'), sep=',')

def visualize_hist(hist, experiment_name, use_batch):
    """
    Plot and save model training loss and accuracy history

    :param hist: model training history object
    :param experiment_name: name of the experiment, for directory saving
    :param use_batch: if 1, show plots in addition to saving
    """
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['accuracy']
    val_acc=hist.history['val_accuracy']
    
    xc=range(len(val_acc))

    plt.figure(3)
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model loss')
    plt.grid(True)
    plt.legend(['Train','Validation'])
    plt.savefig(experiment_name +'/loss.png')
    if use_batch == 1:
        plt.show()

    plt.figure(4)
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy')
    plt.grid(True)
    plt.legend(['Train','Validation','Test'],loc=4)

    plt.savefig(experiment_name +'/acc.png')
    if use_batch==1:
        plt.show()
    
def train_model_lstm(video_set, feature, exp_full_path=None, exp_path=None, val_available=None,
			folder=None, class_file=None, video_dir=None, image_dir=None, image_feature_dir=None, o_flow_dir=None, 
			o_flow_feature_dir=None, csv_file=False, train_path=None, val_path=None, test_path=None, load_model=False, 
            saved_model=None):
    """
    Trains the dynamic motion network

    :param video_set: describes the training set used - usually ELAR
    :param feature: describes the output from the feature extractor
    :param exp_full_path: path to save model training information'
    :param exp_path: N/A
    :param val_available: if true, separate validation set can be used
    :param folder: N/A
    :param class_file: for classes object
    :param video_dir: UNUSED => path to videos?
    :param image_dir: UNUSED = path to images? we're just using features
    :param feature_dir: path to feature directory
    :param o_flow_dir: UNUSED => optical flow directory
    :param o_flow_feature_dir: UNUSED => optical flow feature directory
    :param csv_file: if true, csv file used to construct dataloader
    :param train_path: path to training data
    :param val_path: path to validation data
    :param test_path: path to testing data
    :param load_model: if true, load pretrained sequence model
    :param saved_model: path to saved model 
    """
    if os.path.exists(exp_full_path) == False:
        os.mkdir(exp_full_path, 0o755)

    model_dir = exp_full_path

    print("\nStarting training ...")
    print(os.getcwd())

    # read the classes
    classes_object = VideoClasses(class_file)

    # Image: Load and train the model
    log_path = os.path.join(exp_full_path, 
                            time.strftime("%Y%m%d-%H%M", time.gmtime()) + "-%s%03d-image-mobile-lstm.csv"%(video_set["name"], video_set["classes"]))
    print("Image log: %s" % log_path)

    modelImage = lstm_build(video_set["frames_norm"], feature["output_shape"][0], classes_object._classes_count, 
                            dropout=0.5, model_name=feature["name"])
    
    train_feature_generator(image_feature_dir, model_dir, log_path, modelImage, classes_object,
        batch_size=32, num_epochs=100, learning_rate=1e-4, exp_full_path=exp_full_path, val_available=val_available,
        csv_file=csv_file,train_path=train_path,  val_path=val_path, test_path=test_path, video_set=video_set, 
        feature=feature, load_model=load_model, saved_model=saved_model)

    return
    
    
if __name__ == '__main__':
    video_set = {
        "name" : "ELAR", 
        "classes" : 29,   # number of classes
        "frames_norm" : 18,    # number of frames per video
        "min_dim" : 224,   # smaller dimension of saved video-frames
        "shape" : (224, 224), # height, width
        "fps_avg" : 25,
        "frames_avg" : 18, 
        "duration_avg" : 1.0,# seconds 
        # "transformer":False,
        "reshape_input": False
    }  #True: if the raw input is different from the requested shape for the model

    feature = {
        "name" : "mobilenet",
        "input_shape" : (224, 224, 3),
        "output_shape" : (1024, )
    } # was 1024
	
    data_set_home_path ='./Data/features/mobilenet_temp/'
    class_file_all       = "./DataInfo/ELAR_classes.csv"
    train_path = "./DataInfo/ELAR.csv"
    exp_path = "./mobilenet/20230717_100Epoch_32Batch"
    exp_full_path = os.path.join(os.getcwd(), "results", exp_path)
    image_dir = './Data/images/'

    train_model_lstm(video_set, feature, exp_full_path, exp_path, None, None, class_file_all, 
                     None, None, image_dir, None, None, True, train_path, None, None, False, None)