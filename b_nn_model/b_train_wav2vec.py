# --- Your imports ---
import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from a_nn_metrics import compute_metrics
import preset
from a_prepare_data.a_prep_path import P_devtrain, P_devtest
from a_prepare_data.c_prep_dataset_wav2vec import Wav2VecDataset
from b_nn_model.a_nn_metrics import compute_auc
from b_nn_model.b_nn_model_wav2vec_wrap import EmbeddingModel  # 🛠 注意：换成了EmbeddingModel！


class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        cos_sim = torch.nn.functional.cosine_similarity(x, y, dim=1)
        loss = 1.0 - cos_sim  # 1 - cosine similarity
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# --- Global settings ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 16
epochs = 50
learning_rate = 1e-3
latent_dim = 128
patience = 5
save_dir = preset.dpath_custom_models
os.makedirs(save_dir, exist_ok=True)

machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']

all_eval_results = []

# --- Main training and evaluation loop ---
for machine in machines:
    print(f"\n===== Training Machine: {machine} =====")

    # === Prepare dataset ===
    train_dataset = Wav2VecDataset(part=P_devtrain, machine=machine)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset = Wav2VecDataset(part=P_devtest, machine=machine)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


    # === Initialize Model ===
    model_embed = EmbeddingModel(machine=machine).to(device)
    optimizer = torch.optim.Adam(model_embed.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    criterion = torch.nn.CosineEmbeddingLoss()

    best_auc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model_embed.train()
        total_loss = 0

        # --- Training step ---
        for x_TxF_batch, _, att_AxD_batch in tqdm(train_loader, desc=f"Train-{machine} Epoch {epoch+1}"):
            x_TxF_batch = x_TxF_batch.to(device)
            att_AxD_batch = att_AxD_batch.to(device)

            optimizer.zero_grad()
            x_embed, att_embed = model_embed(x_TxF_batch, att_AxD_batch)

            target = torch.ones(x_embed.size(0), device=device)  # 正样本cosine=1
            loss = criterion(x_embed, att_embed, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- Evaluation step after training ---
        model_embed.eval()
        scores = []
        labels_test = []

        with torch.no_grad():
            for x_TxF_val, y_val, att_val in tqdm(test_loader):
                x_TxF_val = x_TxF_val.to(device)
                att_val = att_val.to(device)

                x_embed, att_embed = model_embed(x_TxF_val, att_val)

                # Anomaly score = 1 - cosine similarity
                cosine_sim = torch.nn.functional.cosine_similarity(x_embed, att_embed, dim=1)
                anomaly_score = 1.0 - cosine_sim  # Higher means more anomalous

                scores.append(anomaly_score.cpu())
                labels_test.append(y_val)

        scores = torch.cat(scores).numpy()
        labels_test = torch.cat(labels_test).numpy()
        print(labels_test)
        print(scores)

        true_labels = (labels_test != 0).astype(int)

        val_auc = compute_auc(true_labels, scores)
        # --- Scheduler Step ---
        scheduler.step(val_auc)

        print(f"→ Epoch {epoch+1} | Train Loss={total_loss:.4f} | Val AUC={val_auc:.4f}")

        # --- Save model if improved ---
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            model_save_path = os.path.join(save_dir, f"EmbeddingModel.{machine}.pth")
            torch.save(model_embed.state_dict(), model_save_path)
            print(f"✅ Embedding Model saved to {model_save_path}")

            # Immediate evaluation CSV save for best model
            eval_results = compute_metrics(scores, true_labels)
            eval_results['Machine'] = machine
            df = pd.DataFrame([eval_results])
            eval_csv_path = os.path.join(save_dir, f"eval.{machine}.csv")
            df.to_csv(eval_csv_path, index=False)
            print(f"✅ Eval metrics saved to {eval_csv_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🔴 Early stopping triggered at epoch {epoch+1}.")
                break

    # Load best eval results after training for merging
    eval_csv_path = os.path.join(save_dir, f"eval.{machine}.csv")
    df = pd.read_csv(eval_csv_path)
    all_eval_results.append(df)

# --- Final merge ---
final_df = pd.concat(all_eval_results, ignore_index=True)
final_eval_path = os.path.join(save_dir, "eval.all.csv")
final_df.to_csv(final_eval_path, index=False)
print(f"\n✅ Final merged evaluation saved to {final_eval_path}")
