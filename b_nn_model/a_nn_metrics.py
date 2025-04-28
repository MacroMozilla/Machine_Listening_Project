# utils_auc.py
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np

def compute_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)


import numpy as np


def compute_pauc(y_true, y_score, p=0.1):
    """
    Compute partial AUC (pAUC) by focusing on the top p fraction of hardest negatives.

    Args:
        y_true (np.ndarray): Binary ground-truth labels (0 or 1).
        y_score (np.ndarray): Predicted scores (continuous).
        p (float): Fraction of negative samples to focus on (e.g., 0.1 for top 10%).

    Returns:
        float: Partial AUC value.
    """

    # Separate positive and negative samples
    scores_pos = y_score[y_true == 1]
    scores_neg = y_score[y_true == 0]

    n_pos = len(scores_pos)
    n_neg = len(scores_neg)

    # Handle edge cases
    if n_pos == 0 or n_neg == 0:
        return np.nan

    # Sort negative scores ascending
    scores_neg_sorted = np.sort(scores_neg)

    # Select top p fraction hardest negatives (lowest scores)
    n_selected_neg = max(int(np.floor(p * n_neg)), 1)
    selected_neg = scores_neg_sorted[:n_selected_neg]

    # Count how many positives rank above the selected negatives
    count = 0
    for s_pos in scores_pos:
        for s_neg in selected_neg:
            if s_pos > s_neg:
                count += 1
            elif s_pos == s_neg:
                count += 0.5

    # Normalize
    pauc = count / (n_pos * n_selected_neg)
    return pauc


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
        'AUC': auc,
        'pAUC': pauc,
        'F1': f1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }
    return results
