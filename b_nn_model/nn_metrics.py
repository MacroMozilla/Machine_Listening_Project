# metrics.py
import numpy as np
from sklearn import metrics


def compute_auc(y_true, y_score):
    """
    Compute AUC based on ground truth and prediction scores.

    Args:
        y_true (array-like): ground truth binary labels (0 for normal, 1 for anomaly)
        y_score (array-like): anomaly scores (higher means more anomalous)

    Returns:
        auc (float): computed AUC score
    """
    return metrics.roc_auc_score(y_true, y_score)


def compute_pauc(y_true, y_score, max_fpr=0.1):
    """
    Compute partial AUC (pAUC) over low FPR range.

    Args:
        y_true (array-like): ground truth binary labels
        y_score (array-like): anomaly scores
        max_fpr (float): maximum false positive rate (default 0.1)

    Returns:
        pauc (float): computed partial AUC
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    if np.max(fpr) < max_fpr:
        # If all FPR are smaller than max_fpr, calculate normal AUC
        return metrics.auc(fpr, tpr) / max_fpr

    mask = fpr <= max_fpr
    fpr = np.concatenate([fpr[mask], [max_fpr]])
    tpr = np.concatenate([tpr[mask], [tpr[mask][-1]]])

    pauc = metrics.auc(fpr, tpr) / max_fpr
    return pauc
