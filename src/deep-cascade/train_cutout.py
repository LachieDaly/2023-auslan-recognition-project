import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from tensorflow import keras


IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
CNN_FEATURES = 512
MEDIAPIPE_FEATURES = 126
ESHR_FEATURES = 75
TOTAL_FEATURES = 2*CNN_FEATURES + MEDIAPIPE_FEATURES + ESHR_FEATURES
BATCH_SIZE = 128
EPOCHS = 10000
LEARNING_RATE = 0.005
MOMENTUM = 0.92

start_time = time.time()

with open('./individual-hand/data.npy', 'rb') as f:
    features = np.load(f)

with open('./recognition-before-resize/labels.npy', 'rb') as f:
    labels = np.load(f)

# with open('./individual-hand/data.npy', 'rb') as f:
#     individual_hand_features = np.load(f)

# with open('./low-confidence/train_data.npy', 'rb') as f:
#     train_featuress = np.load(f)

# with open('./extracted-features/train_masks.npy', 'rb') as f:
#     train_mask = np.load(f)

# with open('./high-confidence/train_labels.npy', 'rb') as f:
#     train_labels = np.load(f)

# with open('./high-confidence/val_data.npy', 'rb') as f:
#     val_features = np.load(f)

# with open('./extracted-features/val_masks.npy', 'rb') as f:
#     val_masks = np.load(f)

# with open('./high-confidence/val_labels.npy', 'rb') as f:
#     val_labels = np.load(f)

# with open('./image-features/train_data.npy', 'rb') as f:
#     train_image_features = np.load(f)


# # with open('./full_image_features/train_labels.npy', 'rb') as f:
# #     train_labels = np.load(f)

# with open('./image-features/val_data.npy', 'rb') as f:
#     val_image_features = np.load(f)


# # with open('./full_image_features/val_labels.npy', 'rb') as f:
# #     val_labels = np.load(f)

with open('./masks/train_masks.npy', 'rb') as f:
    masks = np.load(f)


# with open('./masks/val_masks.npy', 'rb') as f:
#     val_masks = np.load(f)

# print(train_image_features.shape)
# print(val_image_features.shape)

# # train_features[:,:,:512] = train_image_features
# # val_features[:,:,:512] = val_image_features

# # train_features = train_features[:,:,:512]
# # val_features = val_features[:,:,:512]

# # train_features = train_image_features
# # val_features = val_image_features

# inputs = np.concatenate((train_features, val_features), axis=0)
inputs = features
# masks = np.concatenate((train_mask, val_masks), axis=0)
# targets = np.concatenate((train_labels, val_labels), axis=0)
targets = labels

print(inputs.shape)
print(masks.shape)
print(targets.shape)
# print(individual_hand_features.shape)
# # Remove samples with less than 6 frames
msk = np.zeros(shape=(masks.shape[0]), dtype=bool)
for idx, mask in enumerate(masks):
    if np.count_nonzero(mask) < 4:
        msk[idx] = False
    else:
        msk[idx] = True

inputs = inputs[msk]
masks = masks[msk]
targets = targets[msk]


# print(inputs.shape)
# print(masks.shape)
# print(targets.shape)

# # for mask in masks:
# #     if np.count_nonzero(mask) < 7:
# #         print(np.count_nonzero(mask))

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

# # print(train_features.shape)
# # print(val_features.shape)

# # train_data = train_features#, train_mask)

# # val_data = val_features#, val_masks)

train_df = pd.read_csv("C:/Users/Lachie/Desktop/Spreadsheet/train29.csv")


label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
)

def scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 500
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, TOTAL_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    x = keras.layers.LSTM(128, kernel_regularizer=keras.regularizers.L2(0.0001))(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    x = keras.layers.Dense(128, kernel_regularizer=keras.regularizers.L2(0.0001))(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)
    rnn_model = keras.Model([frame_features_input, mask_input], output)

    opt = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    return rnn_model

val_accuracies = []
# Utility for running experiments.
def run_experiment():
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        filepath = "./tmp/gesture_recognition_fold_" + str(fold_no)
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath, save_weights_only=True, save_best_only=True, verbose=1
        )
        early_stop = keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0,
            patience=100,
            verbose=0,
            mode="auto",
            baseline=None,
        )
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        seq_model = get_sequence_model()
        history = seq_model.fit(
            (inputs[train], masks[train]),
            targets[train],
            batch_size=BATCH_SIZE,
            validation_data=((inputs[test], masks[test]), targets[test]),
            epochs=10000,
            callbacks=[checkpoint, early_stop, lr_scheduler],
        )
        end_time = time.time()
        print("Time")
        print(end_time - start_time)
        plt.title("All features - High Confidence")
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['accuracy', 'validation accuracy'])
        plt.savefig('./individual-hand-plots/CKOS_L128_D128_MASK_4/accuracy_fold_' + str(fold_no) + '.png')
        plt.clf()
        seq_model.load_weights(filepath)
        _, accuracy = seq_model.evaluate((inputs[test], masks[test]), targets[test])
        predictions_one_hot = seq_model.predict((inputs[test], masks[test]))
        cm = confusion_matrix(targets[test], predictions_one_hot.argmax(axis=1))
        classes = ('Arrive', 'Bed', 'Bird', 'Boy', 'Come', 'Day', 'Deer', 'Frog', 'Girl', 'Good', 'Lady', 'Laugh', 'Man', 
            'Night', 'People', 'Rabbit', 'Real', 'Same', 'Say', 'Sheep', 'Slow', 'Sprint', 'Think', 'Tortoise', 'What', 
            'Where', 'Window', 'Wolf', 'Yell')
        df_cm = pd.DataFrame(cm, index = [i for i in classes],
                        columns = [i for i in classes])
        
        plt.figure(figsize = (12,7))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        sn.heatmap(df_cm, annot=True)
        plt.savefig('./individual-hand-plots/CKOS_L128_D128_MASK_4/confusion_fold_' + str(fold_no) + '.png')
        plt.clf()
        val_accuracies.append(accuracy)
        fold_no = fold_no + 1


run_experiment()

for val_accuracy in val_accuracies:
    print(f"Test accuracy: {round(val_accuracy * 100, 2)}%")