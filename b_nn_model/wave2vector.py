# --- Imports ---
from a_prepare_data.a_prep_path import P_devtrain, P_devtest
from a_prepare_data.c_prep_dataset_wav2vec import Wav2VecDataset
from f_utility.plot_tools import plt_img, plt_show

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from nn_metrics import compute_auc, compute_pauc

import numpy as np
from tqdm import tqdm

# --- Settings ---
device = 'cuda'
machine = 'bearing'
parts = [P_devtrain, P_devtest]
is_train_flags = [1, 0]

# --- Containers ---
all_features = []
all_labels = []
all_is_train = []

# --- Load features ---
for part, is_train_flag in zip(parts, is_train_flags):
    dataset = Wav2VecDataset(part=part, machine=machine)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            x_TxF_batch, y_B, _ = batch  # [B, T, F]
            x_BxF = x_TxF_batch.mean(dim=1)  # Mean pool along T → [B, F]
            all_features.append(x_BxF.cpu())
            all_labels.append(y_B)
            all_is_train.append(torch.full((x_BxF.size(0),), is_train_flag))

# --- Stack all data ---
all_features = torch.cat(all_features, dim=0)  # [N, F]
all_labels = torch.cat(all_labels, dim=0)
all_is_train = torch.cat(all_is_train, dim=0)

# --- Split train/test ---
train_mask = all_is_train == 1
test_mask = all_is_train == 0

features_train = all_features[train_mask]  # [n_train, F]
labels_train = all_labels[train_mask]

features_test = all_features[test_mask]    # [n_test, F]
labels_test = all_labels[test_mask]

# --- Normalize (StandardScaler) ---
scaler = StandardScaler()
features_train_norm = torch.tensor(
    scaler.fit_transform(features_train.numpy()), dtype=torch.float32
)
features_test_norm = torch.tensor(
    scaler.transform(features_test.numpy()), dtype=torch.float32
)

# --- Optional PCA (very minor, usually good) ---
# pca = PCA(n_components=32)  # Uncomment if you want to try PCA
# features_train_norm = torch.tensor(pca.fit_transform(features_train_norm.numpy()))
# features_test_norm = torch.tensor(pca.transform(features_test_norm.numpy()))

# --- Build kNN model (打榜公认最好的是 kNN Distance based anomaly score) ---
n_neighbors = 5
knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
knn_model.fit(features_train_norm.numpy())

# --- Compute anomaly scores ---
distances, _ = knn_model.kneighbors(features_test_norm.numpy())  # [n_test, n_neighbors]
scores = distances.mean(axis=1)  # k个最近邻的平均距离作为anomaly score

# --- Ground truth (0=normal, 1=anomaly) ---
true_test = (labels_test != 0).long()

# --- Metrics ---
auc = compute_auc(true_test.numpy(), scores)
pauc = compute_pauc(true_test.numpy(), scores, max_fpr=0.1)

# --- Find best threshold for F1 ---
thresholds = torch.linspace(scores.min(), scores.max(), steps=500)
best_f1 = -1
best_thresh = None

for thresh in thresholds:
    preds = (torch.tensor(scores) > thresh).long()
    f1 = f1_score(true_test.numpy(), preds.numpy())
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh.item()

# --- Apply best threshold ---
preds_test = (torch.tensor(scores) > best_thresh).long()

# --- Final Evaluation ---
tn, fp, fn, tp = confusion_matrix(true_test.numpy(), preds_test.numpy()).ravel()

precision = precision_score(true_test.numpy(), preds_test.numpy())
recall = recall_score(true_test.numpy(), preds_test.numpy())
f1 = f1_score(true_test.numpy(), preds_test.numpy())
accuracy = accuracy_score(true_test.numpy(), preds_test.numpy())

# --- Print Results ---
print("\nConfusion Matrix:")
print(f"{'':>10} {'Pred 0':>10} {'Pred 1':>10}")
print(f"{'True 0':>10} {tn:10} {fp:10}")
print(f"{'True 1':>10} {fn:10} {tp:10}")

print("\nMetrics:")
print(f"AUC   : {auc:.4f}")
print(f"pAUC  : {pauc:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# --- Plot (optional) ---
plt_show()
