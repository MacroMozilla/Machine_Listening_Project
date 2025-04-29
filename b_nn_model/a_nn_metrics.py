# utils_auc.py
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np

def compute_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)


def compute_pauc(y_true, y_score, max_fpr=0.1):
    return roc_auc_score(y_true, y_score, max_fpr=max_fpr)


# --- Metrics computation function ---
def compute_metrics(pred_scores, true_labels):
    """
    pred_scores: numpy array, shape (N,)
    true_labels: numpy array, shape (N,)
    """

    assert pred_scores.shape == true_labels.shape, "Shape mismatch between pred_scores and true_labels"

    auc = compute_auc(true_labels, pred_scores)
    pauc = compute_pauc(true_labels, pred_scores)

    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
    f1_scores = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr) + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    preds = (pred_scores > best_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    accuracy = accuracy_score(true_labels, preds)

    results = {
        'AUC': round(auc,4),
        'pAUC': round(pauc,4),
        'F1': round(f1,4),
        'Accuracy': round(accuracy,4),
        'Precision': round(precision,4),
        'Recall': round(recall,4)
    }
    return results
