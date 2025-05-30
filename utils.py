import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def clf_report_to_df(clf_report, clf_scores_path=None, clf_per_class_scores_path=None):
    '''
    Converts a classification report dictionary to two pandas DataFrames:
    1. A DataFrame containing overall classification scores (accuracy, f1-score, precision, recall).
    2. A DataFrame containing per-class classification scores (f1-score, precision, recall) for each class.
    If paths are provided, the DataFrames will be saved as CSV files.

    Parameters:
    clf_report (dict): The classification report dictionary from sklearn.metrics.classification_report.
    clf_scores_path (str, optional): Path to save the overall classification scores DataFrame as a CSV file.
    clf_per_class_scores_path (str, optional): Path to save the per-class classification scores DataFrame as a CSV file.

    Returns:
    clf_scores_df (pd.DataFrame): DataFrame containing overall classification scores.
    per_class_clf_scores_df (pd.DataFrame): DataFrame containing per-class classification scores.
    '''
    
    clf_scores = {}
    clf_scores['accuracy'] = clf_report['accuracy']
    clf_scores['f1-score macro avg'] = clf_report['macro avg']['f1-score']
    clf_scores['f1-score weighted avg'] = clf_report['weighted avg']['f1-score']
    clf_scores['precision macro avg'] = clf_report['macro avg']['precision']
    clf_scores['precision weighted avg'] = clf_report['weighted avg']['precision']
    clf_scores['recall macro avg'] = clf_report['macro avg']['recall']
    clf_scores['recall weighted avg'] = clf_report['weighted avg']['recall']
    clf_scores_df = pd.DataFrame([clf_scores])

    if clf_scores_path:
        clf_scores_df.to_csv(clf_scores_path, index=False)

    for key in ['accuracy', 'macro avg', 'weighted avg']:
        del clf_report[key]
    per_class_clf_scores_df = pd.DataFrame(clf_report).T

    if clf_per_class_scores_path:
        per_class_clf_scores_df.to_csv(clf_per_class_scores_path)

    return clf_scores_df, per_class_clf_scores_df

def plot_confusion_matrix(y_true, y_pred, title, path=None):
    '''
    Plots a confusion matrix for the given true and predicted labels.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    title (str): Title for the confusion matrix plot.
    path (str, optional): Path to save the confusion matrix plot as an image file. If None, the plot will be shown but not saved.
    '''
    
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()