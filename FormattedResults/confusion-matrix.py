import matplotlib.pyplot as plt
import seaborn as sn
import csv
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def save_confusion_matrix(y_true, y_pred):
    classes = ('Arrive', 'Bed', 'Bird', 'Boy', 'Come', 'Day', 'Deer', 'Frog', 'Girl', 'Good', 'Lady', 'Laugh', 
               'Man', 'Night', 'People', 'Rabbit', 'Real', 'Same', 'Say', 'Sheep', 'Slow', 'Sprint', 'Think', 
               'Tortoise', 'What', 'Where', 'Window', 'Wolf', 'Yell')
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('ResNet-101_confusion-matrix.png')

def save_classification_report(y_true, y_pred):
    classes = ['Arrive', 'Bed', 'Bird', 'Boy', 'Come', 'Day', 'Deer', 'Frog', 'Girl', 'Good', 'Lady', 'Laugh', 
               'Man', 'Night', 'People', 'Rabbit', 'Real', 'Same', 'Say', 'Sheep', 'Slow', 'Sprint', 'Think', 
               'Tortoise', 'What', 'Where', 'Window', 'Wolf', 'Yell']
    cr = classification_report(y_true, y_pred, target_name=classes)
    file = open("")


if __name__ == 'main':
    prediction_labels_csv = "something.csv"
    true_labels_csv = "something.csv"

    experiment_name = "something"
    predict_df = pd.read_csv(prediction_labels_csv)
    predict_labels = predict_df[predict_df.columns[1]]
    true_df = pd.read_csv(true_labels_csv)
    true_labels = true_df[true_df.columns[1]]

    save_classification_report()