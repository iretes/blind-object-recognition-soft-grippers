import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

def get_trials_df(tuner, objective='score'):
    trials = tuner.oracle.trials.values()
    hps = []
    for trial in trials:
        hp_values = trial.hyperparameters.get_config()["values"]
        hp_values[objective] = trial.score
        hps.append(hp_values)
    hp_df = pd.DataFrame(hps)
    hp_df = hp_df.sort_values(by=objective)
    return hp_df.style.background_gradient(cmap="Blues", subset=[objective])

def get_clf_report_dfs(
        y_true, y_pred,
        clf_scores_path=None,
        clf_per_class_scores_path=None
    ):
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

def plot_confusion_matrix(y_true, y_pred, title, path=None, labels=None):
    if labels is None:
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