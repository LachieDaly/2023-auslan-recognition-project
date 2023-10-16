import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def save_confusion_matrix(y_true, y_pred, experiment_name):
    classes = ('Arrive', 'Bed', 'Bird', 'Boy', 'Come', 'Day', 'Deer', 'Frog', 'Girl', 'Good', 'Lady', 'Laugh', 
               'Man', 'Night', 'People', 'Rabbit', 'Real', 'Same', 'Say', 'Sheep', 'Slow', 'Sprint', 'Think', 
               'Tortoise', 'What', 'Where', 'Window', 'Wolf', 'Yell')
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('./FormattedResults/' + experiment_name + '.png')

def save_classification_report(y_true, y_pred, experiment_name):
    classes = ['Arrive', 'Bed', 'Bird', 'Boy', 'Come', 'Day', 'Deer', 'Frog', 'Girl', 'Good', 'Lady', 'Laugh', 
               'Man', 'Night', 'People', 'Rabbit', 'Real', 'Same', 'Say', 'Sheep', 'Slow', 'Sprint', 'Think', 
               'Tortoise', 'What', 'Where', 'Window', 'Wolf', 'Yell']
    cr = classification_report(y_true, y_pred, target_names=classes)
    with open("./FormattedResults/" + experiment_name + ".txt", "w") as text_file:
        text_file.write(cr)


if __name__ == "__main__":
    prediction_labels_csv = "./FormattedResults/3dcnn/3dcnn_predictions.csv"
    true_labels_csv = "./FormattedResults/predictions.csv"

    experiment_name = prediction_labels_csv.split("/")[-1].replace(".csv", "")
    predict_df = pd.read_csv(prediction_labels_csv)
    predict_labels = predict_df[predict_df.columns[1]].values.tolist()
    true_df = pd.read_csv(true_labels_csv)
    true_labels = true_df[true_df.columns[1]].values.tolist()

    save_confusion_matrix(true_labels, predict_labels, experiment_name)
    save_classification_report(true_labels, predict_labels, experiment_name)