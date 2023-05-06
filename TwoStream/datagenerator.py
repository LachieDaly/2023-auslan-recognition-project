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


from frame import files_to_frames, process_images, frames_show, images_normalize_withGrayscale

class FramesGenerator(tf.keras.utils.Sequence):
    """
    Read and yields video frames/optical flow for Keras.model.fit_generator
    Generator can be used for multi-threading
    Substantial initialisation and checks upfront, including one-hot-encoding of labels
    """
    def __init__(self, path:str, batch_size:int, frames:int, height: int, width:int, channels:int, \
        classes:list = None, shuffle:bool = True, resize:bool = False, convert_to_grapy:bool = False):
        """
        Assum directory structures: --this isn't our structure currently, but it could be
        ... / path / class / videoname / frames.jpg
        """
        'Initialise'
        self._batch_size = batch_size
        self._frames = frames
        self._height = height
        self._width = width
        self._channels = channels
        self._shape = (frames, height, width, channels)
        self._shuffle = shuffle
        self._resize = resize
        self._convert_to_graph = convert_to_grapy

        # retrieve all videos 
        self._videos = pd.DataFrame(sorted(glob.glob(path + "/*/*")), columns=["frame_dir"])
        self._sample_count = len(self._videos)
        if self._sample_count == 0:
            raise ValueError("Found no frame directories files in " + path)
        
        print("Detected %d samples in %s ..." % (self._sample_count, path))

        # extract (text) labels from path
        labels = self._videos.frame_dir.apply(lambda s: os.path.normpath(s).split("\\")[-2])
        self._videos.loc[:, "label"] = labels

        # extract unique classes from all detected labels
        self._classes = sorted(list(self._videos.label.unique()))

        # if classes are provided upfront
        if classes != None:
            classes = sorted(np.unique(classes))
            # check detected vs provided classes
            if set(self._classes).issubset(set(classes)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")

            self._classes = classes

        self._classes_count = len(self._classes)

        # encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(self._classes)
        self._videos.loc[:, "label"] = label_encoder.transform(self._videos.label)

        self.on_epoch_end()
        return

    def __len__(self):
        """
        Denotes the number of gbatches per epoch
        """
        return int(np.ceil(self._sample_count / self._batch_size))

    def __getitem__(self, step):
        """
        Generate one batch of data
        """

        # Generate indexes of the batch
        indexes = self.indexes[step*self._batch_size]

        videos_batch = self._videos.loc[indexes, :]
        batch_size = len(videos_batch)

        x = np.empty((batch_size, ) + self._shape, dtype = float)
        y = np.empty((batch_size), dtype = int)

        # Generate data
        for i in range(batch_size):
            x[i,], y[i] = self.__data_generation(videos_batch.iloc[i, :])

        return x, np_utils.to_categorical(y, num_classes=self._classes_count)

    def __data_generation(self, video:pd.Series):
        """
        Returns frames for 1 video, including normalising & preprocessing
        """
        frames = files_to_frames(video.frame_dir)

        frames = frames[..., 0:self._channels]

        frames = process_images(frames, self._frames, self._height, self._width, rescale = True)

        return frames, video.label

    def data_generation(self, video:pd.Series):
        return self.__data_generation(video)


    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self._indexes = np.arange(self._sample_count)
        if self._shuffle == True:
            np.random.suffle(self._indexes)

class FramesGenerator_withSplitting(tf.keras.utils.Sequence):
    """
    Read and yields video frames/optical flow for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    """

    def __init__(self, path:str, \
        batch_size:int, frames:int, height:int, width:int, channels:int, \
        classes_full:list = None, shuffle:bool = True, resize:bool = False, convert_to_graph:bool = False):
        """
        Assume directory structure:
        ... / sPath / class / videoname / frames.jpg
        """

        'Initialization'
        self._batch_size = batch_size
        self._frames = frames
        self._height = height
        self._width = width
        self._channels = channels
        self._shape = (frames, height, width, channels)
        self._shuffle = shuffle
        self._resize = resize
        self._convert_to_graph = convert_to_graph
        		
        # retrieve all videos = frame directories
        self._videos = pd.DataFrame(sorted(glob.glob(path + "/*/*")), columns=["frame_dir"])
        self._sample_count = len(self._videos)
        if self._sample_count == 0: 
            raise ValueError("Found no frame directories files in " + path)

        # extract (text) labels from path
        labels =  self._videos.frame_dir.apply(lambda s: s.split("/")[-2])
        self._videos.loc[:, "label"] = labels
            
        # extract unique classes from all detected labels
        self._classes = sorted(list(self._videos.label.unique()))

        # if classes are provided upfront
        if classes_full != None:
            classes_full = sorted(np.unique(classes_full))
            # check detected vs provided classes
            if set(self._classes).issubset(set(classes_full)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            self._classes = classes_full
            
        self._classes_count = len(self._classes)

        # encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(self._classes)
        self._videos.loc[:, "label"] = label_encoder.transform(self._videos.label)
        
        self.on_epoch_end()
        return

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(self._sample_count / self._batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self._indexes = np.arange(self._sample_count)
        if self._shuffle == True:
            np.random.shuffle(self._indexes)


    def __getitem__(self, step_count):
        """
        Generate one batch of data
        """

        # Generate indexes of the batch
        indexes = self._indexes[step_count*self._batch_size:(step_count+1)*self._batch_size]

        # get batch of videos
        videos_batch = self._videos.loc[indexes, :]
        batch_size = len(videos_batch)

        # initialize arrays
        x = np.empty((batch_size, ) + self._shape, dtype = float)
        y = np.empty((batch_size), dtype = int)

        # Generate data
        for i in range(batch_size):
            x[i,], y[i] = self.__data_generation(videos_batch.iloc[i,:])

        # onehot the labels
        return x, np_utils.to_categorical(y, num_classes=self._classes_count)


    def __data_generation(self, video:pd.Series):
        """
        Returns frames for 1 video, including normalizing & preprocessing
        """
       
        # Get the frames from disc
        frames = files_to_frames(video.frame_dir, self._convert_to_graph, self._resize)

        # only use the first nChannels (typically 3, but maybe 2 for optical flow)
        frames = frames[..., 0:self._channels]
		
        frames = images_normalize_withGrayscale(frames, self._frame_count, self._height, self._width, rescale = True)
        
        return frames, video.label

    def data_generation(self, video:pd.Series):
        return self.__data_generation(video)

class FeaturesGenerator_multi_withSplitting(tf.keras.utils.Sequence):
    """
    Read and yields video frames/optical flow for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    """

    def __init__(self, path_one:str, path_two:str, batch_size:int, shape_one, shape_two, \
        classes_full:list = None, shuffle:bool = True):
        """
        Assume directory structure:
        ... / sPath_01 / class / videoname / frames.jpg
        """

        'Initialization'
        self._batch_size = batch_size
        self._shape_one = shape_one
        self._shape_two = shape_two
        self._shuffle = shuffle
        		
        # retrieve all videos = frame directories
        self._samples_df_one, self._sample_count_one, self._classes_one, self._samples_one = self.initialize_gen(path_one, classes_full, shape_one)
        self._samples_df_two, self._sample_count_two, self._classes_two, self._samples_two = self.initialize_gen(path_two, classes_full, shape_two)
        
        self.on_epoch_end()
        return

    def initialize_gen(self, path, classes_full, shape):
        samples_df = path
        sample_count = len(samples_df)
        if sample_count == 0: raise ValueError("Found no feature files in " + path)
        print("Detected %d samples in %s ..." % (sample_count, path))

        # test shape of first sample
        x = np.load(samples_df.path[0])
        if x.shape != shape: raise ValueError("Wrong feature shape: " + str(x.shape) + str(shape))

        # extract (text) labels from path
        labels =  samples_df.path.apply(lambda s: s.split("/")[-2])
        samples_df.loc[:, "label"] = labels.astype(int) #hamzah: I added   .astype(int)       
           
        # extract unique classes from all detected labels
        classes = sorted(list(samples_df.sLabel.unique()))
        #hamzah
        classes = [int(i) for i in classes]
        #print(self.liClasses)		

        # if classes are provided upfront
        if classes_full != None:
            classes_full = sorted(np.unique(classes_full).astype(int))
            # check detected vs provided classes
            if set(classes).issubset(set(classes_full)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            classes = classes_full
            
        classes_count = len(classes_count)

        # encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(classes)
        samples_df.loc[:, "label"] = label_encoder.transform(samples_df.label)

        return samples_df,  classes_count, classes, sample_count
        
    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(self._sample_count_one / self._batch_size))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self._indexes = np.arange(self._sample_count_one)
        if self._shuffle == True:
            np.random.shuffle(self._indexes)


    def __getitem__(self, step_count):
        """
        Generate one batch of data
        """

        # Generate indexes of the batch
        indexes = self._indexes[step_count*self._batch_size:(step_count+1)*self._batch_size]

        # get batch of videos
        samples_batch_one = self._sample_df_one.loc[indexes, :]
        samples_batch_two = self._sample_df_two.loc[indexes, :]

        batch_size = len(samples_batch_one)

        # initialize arrays
        x_one = np.empty((batch_size, ) + self._shape_one, dtype = float)
        x_two = np.empty((batch_size, ) + self._shape_two, dtype = float)

        y = np.empty((batch_size), dtype = int)
        # y_two = np.empty((batch_size), dtype = int)

        # Generate data
        for i in range(batch_size):
            # generate data for single video(frames)
            x_one[i,], y[i] = self.__data_generation(samples_batch_one.iloc[i,:])
            x_two[i,], y[i] = self.__data_generation(samples_batch_two.iloc[i,:])
            #print("Sample #%d" % (indexes[i]))

        # onehot the labels
        x = [x_one, x_two]
        return x, keras.utils.np_utils.to_categorical(y, num_classes=self._classes_one)


    def __data_generation(self, sample:pd.Series):
        """
        Returns frames for 1 video, including normalizing & preprocessing
        """

        try:
            x = np.load(sample.path)

            return x, sample.label
        except:
            print(sample.path)
            raise ValueError("Error "+ sample.path)

class FeaturesGenerator_multiInput(tf.keras.utils.Sequence):
    """
    return samples from two sources and check if they match
    """

    def __init__(self, path_one:str, path_two:str, batch_size:int, shape_one, shape_two, \
        classes_full:list = None, shuffle:bool = True):
        """
        Assume directory structure:
        ... / sPath_01 / class / videoname / frames.jpg
        """

        'Initialization'
        self._batch_size = batch_size
        self._shape_one = shape_one
        self._shape_two = shape_two
        self._shuffle = shuffle
        self._path_two = path_two
        		
        # retrieve all videos = frame directories
        self._samples_df_one, self._classes_count_one, self._classes_one, self._sample_count_one = self.initialize_gen(path_one, classes_full, shape_one)
        
        self.on_epoch_end()
        return

    def initialize_gen(self, path, classes_full, shape):
                #print(sPath )
        samples_df = path
        sample_count = len(samples_df)
        if sample_count == 0: 
            raise ValueError("Found no feature files in " + path)
        print("Detected %d samples in %s ..." % (sample_count, path))

        # test shape of first sample
        arX = np.load(samples_df.path[0])
        if arX.shape != shape: 
            raise ValueError("Wrong feature shape: " + str(arX.shape) + str(shape))

        # extract (text) labels from path
        labels =  samples_df.sPath.apply(lambda s: s.split("/")[-2])
        samples_df.loc[:, "label"] = labels.astype(int) #hamzah: I added   .astype(int)       
           
        # extract unique classes from all detected labels
        classes = sorted(list(samples_df.label.unique()))
        #hamzah
        classes = [int(i) for i in classes]
        #print(liClasses)		

        # if classes are provided upfront
        if classes_full != None:
            classes_full = sorted(np.unique(classes_full).astype(int))
            # check detected vs provided classes
            if set(classes).issubset(set(classes_full)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            classes = classes_full
            
        classes_count = len(classes)

        # encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(classes)
        samples_df.loc[:, "label"] = label_encoder.transform(samples_df.label)

        return samples_df, classes_count, classes, sample_count
        
    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(self._sample_count_one / self._batch_size))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self._indexes = np.arange(self._sample_count_one)
        if self._shuffle == True:
            np.random.shuffle(self._indexes)


    def __getitem__(self, step_count):
        """
        Generate one batch of data
        """

        # Generate indexes of the batch
        indexes = self.indexes[step_count*self._batch_size:(step_count+1)*self._batch_size]

        # get batch of videos
        sample_batch_one = self._samples_df_one.loc[indexes, :]

        batch_size = len(sample_batch_one)

        # initialize arrays
        x_one = np.empty((batch_size, ) + self._shape_one, dtype = float)
        x_two = np.empty((batch_size, ) + self._shape_two, dtype = float)

        y = np.empty((batch_size), dtype = int)

        # Generate data
        for i in range(batch_size):
          
            x_one[i,], y[i] = self.__data_generation(sample_batch_one.iloc[i,:])

            filename_one = os.path.basename(sample_batch_one.iloc[i,:].path)
            sign_id = sample_batch_one.iloc[i,:].path.split('/')
            fileName_02 = os.path.join(self._path_two, sign_id[-2], filename_one.replace('npy','jpg'))

            x_two[i,], y[i] = self.__data_generation_v2(fileName_02, sample_batch_one.iloc[i,:].label)

        # onehot the labels
        x = [x_one, x_two]
        return x, keras.utils.np_utils.to_categorical(y, num_classes=self._classes_count_one)


    def __data_generation(self, sample:pd.Series):
        """
        Returns frames for 1 video, including normalizing & preprocessing
        """
       
        # Get the frames from disc
        'Generates data for 1 sample' 
        
        'Generates data for 1 sample' 
        try:
            x = np.load(sample.path)

            return x, sample.label
        except:
            print(sample.path)
            raise ValueError("Error "+ sample.path)

    def __data_generation_v2(self, filepath, label):
        """
        Returns frames for 1 video, including normalizing & preprocessing
        """
       
        # Get the frames from disc
        'Generates data for 1 sample' 
        
        'Generates data for 1 sample' 
        try:
            #arX = np.load(filePath)
            x = cv2.imread(filepath)
            x = cv2.resize(x, self._shape_two[0:2])
            
            return x, label
        except Exception as e:
            #print(filePath)
            print(e)
            print(self._shape_two[0:2])
            raise ValueError("Error " + filepath)


class FeaturesGenerator(tf.keras.utils.Sequence):
    """
    Reads and yields (preprocessed) I3D features for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    """

    def __init__(self, path:str, batch_size:int, shape, \
        classes_full:list = None, shuffle:bool = True):
        """
        Assume directory structure:
        ... / sPath / class / feature.npy
        """

        'Initialization'
        self._batch_size = batch_size
        self._shape = shape
        self._shuffle = shuffle

        # retrieve all feature files
        self._samples_df = pd.DataFrame(sorted(glob.glob(path + "/*/*.npy")), columns=["path"])
        self._sample_count = len(self._samples_df)
        if self._sample_count == 0: 
            raise ValueError("Found no feature files in " + path)
        print("Detected %d samples in %s ..." % (self._sample_count, path))

        # test shape of first sample
        x = np.load(self._samples_df.path[0])
        if x.shape != shape: 
            raise ValueError("Wrong feature shape: " + str(x.shape) + str(shape))

        # extract (text) labels from path
        labels =  self._samples_df.path.apply(lambda s: s.split("/")[-2])
        self._samples_df.loc[:, "label"] = labels.astype(int) #hamzah: I added   .astype(int)       
        #print(self.dfSamples.loc[:, "sLabel"])		
           
        # extract unique classes from all detected labels
        self._classes = sorted(list(self._samples_df.label.unique()))
        #hamzah
        self._classes = [int(i) for i in self._classes]
        #print(self.liClasses)		

        # if classes are provided upfront
        if classes_full != None:
            classes_full = sorted(np.unique(classes_full).astype(int))
            #print(liClassesFull)
            # check detected vs provided classes
            if set(self._classes).issubset(set(classes_full)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            self._classes = classes_full
            
        self._classes_count = len(self._classes)

        # encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(self._classes)
        self._samples_df.loc[:, "label"] = label_encoder.transform(self._samples_df.sLabel)
        print('in Initalization');
        
        self.on_epoch_end()
        return

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(self._sample_count / self._batch_size))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self._indexes = np.arange(self._sample_count)
        if self._shuffle == True:
            np.random.shuffle(self._indexes)

    def __getitem__(self, step_count):
        """
        Generate one batch of data
        """

        # Generate indexes of the batch
        indexes = self.indexes[step_count*self._batch_size:(step_count+1)*self._batch_size]

        # Find selected samples
        samples_batch = self._samples_df.loc[indexes, :]
        batch_size = len(samples_batch)

        # initialize arrays
        x = np.empty((batch_size, ) + self._shape, dtype = float)
        y = np.empty((batch_size), dtype = int)

        # Generate data
        for i in range(batch_size):
            # generate single sample data
            x[i,], y[i] = self.__data_generation(samples_batch.iloc[i,:])

        # onehot the labels
        return x, keras.utils.to_categorical(y, num_classes=self._classes_count)

    def __data_generation(self, seSample:pd.Series):
        """
        Generates data for 1 sample
        """

        x = np.load(seSample.path)

        return x, seSample.label

class FeaturesGenerator_withSplitting(tf.keras.utils.Sequence):
    """
    Reads and yields (preprocessed) I3D features for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    """

    def __init__(self, path:str, batch_size:int, shape, \
        classes_full:list = None, shuffle:bool = True, video_set=None, feature = None):
        """
        Assume directory structure:
        ... / sPath / class / feature.npy
        """

        'Initialization'
        self._batch_size = batch_size
        self._shape = shape
        self._feature = feature
        self._shuffle = shuffle
        self._video_set = video_set
        self._new_input_shape = (self._video_set["frames_norm"], self._feature["input_shape"][0],self._feature["input_shape"][1],self._feature["input_shape"][2])
        # retrieve all feature files
        self._samples_df = path
        self._sample_count = len(self._samples_df)
        if self._sample_count == 0: 
            raise ValueError("Found no feature files in " + path)
        print("Detected %d samples in %s ..." % (self._sample_count, path))

        # test shape of first sample
        x = np.load(self._samples_df.path[0])
        if self._video_set != None and self._video_set["reshape_input"] == True: 
            x = x.reshape(self._new_input_shape)
        else:
            if x.shape != shape:
                raise ValueError("Wrong feature shape: " + str(x.shape) + str(shape))

        # extract (text) labels from path
        labels =  self._samples_df.path.apply(lambda s: s.split("/")[-2])
        self._samples_df.loc[:, "label"] = labels.astype(int) #hamzah: I added   .astype(int)       
        #print(self.dfSamples.loc[:, "sLabel"])		
           
        # extract unique classes from all detected labels
        self._classes = sorted(list(self._samples_df.sLabel.unique()))
        #hamzah
        self._classes = [int(i) for i in self._classes]
        #print(self.liClasses)		

        # if classes are provided upfront
        if classes_full != None:
            classes_full = sorted(np.unique(classes_full).astype(int))
            if set(self._classes).issubset(set(classes_full)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            self._classes = classes_full
            
        self._class_count = len(self._classes)

        # encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(self._classes)
        self._samples_df.loc[:, "label"] = label_encoder.transform(self._sample_df.label)
        
        self.on_epoch_end()
        return

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(self._sample_count / self._batch_size))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self._indexes = np.arange(self._sample_count)
        if self._shuffle == True:
            np.random.shuffle(self._indexes)

    def __getitem__(self, step_count):
        """
        Generate one batch of data
        """

        # Generate indexes of the batch
        indexes = self.indexes[step_count*self._batch_size:(step_count+1)*self._batch_size]

        # Find selected samples
        sample_batch = self._samples_df.loc[indexes, :]
        batch_size = len(sample_batch)

        # initialize arrays
        x = np.empty((batch_size, ) + self._shape, dtype = float)
        y = np.empty((batch_size), dtype = int)

        # Generate data
        for i in range(batch_size):
            # generate single sample data
            x[i,], y[i] = self.__data_generation(sample_batch.iloc[i,:])
        #print(arX.shape)
        # onehot the labels
        return x, tf.keras.utils.to_categorical(y, num_classes=self._class_count)

    def __data_generation(self, sample:pd.Series):
        'Generates data for 1 sample' 
        
        'Generates data for 1 sample' 
        try:
            #print(seSample.sPath)
            x = np.load(sample.path)
            #print(arX.shape)
            if self._video_set != None and self._video_set["reshape_input"] == True: 
                x = x.reshape(self._new_input_shape) 

            return x, sample.label
        except:
            print(sample.path)
            raise ValueError("Error "+ sample.sPath)


class VideoClasses():
    """
    Loads the video classes (incl descriptions) from a csv file
    """
    def __init__(self, class_file:str):
        # load label description: index, sClass, sLong, sCat, sDetail
        self._class_df = pd.read_csv(class_file)

        # sort the classes
        self._class_df = self._class_df.sort_values("s_class").reset_index(drop=True)
        
        self._classes = list(self._class_df.s_class)
        self._classes_count = len(self._class_df)

        print("Loaded %d classes from %s" % (self.nClasses, class_file))
        return
