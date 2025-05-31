import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

def get_trials_df(tuner):
    '''
    Converts the trials from a Keras Tuner object into a pandas DataFrame.

    Parameters:
    tuner (kerastuner.Tuner): A Keras Tuner object containing the trials.
    
    Returns:
    pd.DataFrame: A DataFrame containing the hyperparameter values and
        validation loss for each trial.
    '''
    
    trials = tuner.oracle.trials.values()
    hps = []
    for trial in trials:
        hp_values = trial.hyperparameters.get_config()["values"]
        hp_values["val_loss"] = trial.score
        hps.append(hp_values)
    hp_df = pd.DataFrame(hps)
    hp_df = hp_df.sort_values(by="val_loss")
    return hp_df.style.background_gradient(cmap="Blues", subset=["val_loss"])

def get_clf_report_dfs(
        y_true, y_pred,
        clf_scores_path=None,
        clf_per_class_scores_path=None
    ):
    '''
    Generates classification report DataFrames from true and predicted labels.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    clf_scores_path (str, optional): Path to save the overall classification
        scores as a CSV file. If None, scores are not saved.
    clf_per_class_scores_path (str, optional): Path to save the per-class
        classification scores as a CSV file. If None, scores are not saved.
    
    Returns:
    tuple: A tuple containing two DataFrames:
        - Overall classification scores DataFrame.
        - Per-class classification scores DataFrame.
    '''

    clf_report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dict=True
    )

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
    path (str, optional): Path to save the confusion matrix plot as an image
        file. If None, the plot will be shown but not saved.
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