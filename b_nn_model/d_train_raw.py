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
from a_prepare_data.b_prep_dataset import WavDataset
from b_nn_model.a_nn_metrics import compute_auc
from b_nn_model.b_nn_loss import CosineSimilarity01, matrix_bce_loss_balanced_v2, cosine_similarity_matrix
from b_nn_model.c_nn_model import EmbeddingModel, RawFeatureExtractor  # 🛠 注意：换成了EmbeddingModel！


# --- Global settings ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 20
epochs = 50
learning_rate = 1e-3
latent_dim = 64
patience = 5
save_dir = os.path.join(preset.dpath_custom_models, WavDataset.__name__)
os.makedirs(save_dir, exist_ok=True)

machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve'][:]
all_eval_results = []

# --- Main loop ---
for machine in machines:
    print(f"\n===== Training Machine: {machine} =====")

    # Prepare datasets
    train_dataset = WavDataset(part=P_devtrain, machine=machine)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = WavDataset(part=P_devtest, machine=machine)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model, optimizer, etc.
    model_embed = EmbeddingModel(machine=machine,F_in=16000,feature_extractor=RawFeatureExtractor).to(device)
    optimizer = torch.optim.Adam(model_embed.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience, factor=0.5)
    cosim01 = CosineSimilarity01()

    best_auc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model_embed.train()

        train_cosim_s = []
        N = len(train_loader.dataset)
        for x_BxTxF, _, att_BxAxD in tqdm(train_loader, desc=f"Train-{machine} Epoch {epoch + 1}"):
            x_BxTxF, att_BxAxD = x_BxTxF.to(device), att_BxAxD.to(device)

            B, T, F = x_BxTxF.shape

            optimizer.zero_grad()
            x_embed, att_embed = model_embed(x_BxTxF, att_BxAxD)

            cosim_B = cosim01(x_embed, att_embed)

            loss = matrix_bce_loss_balanced_v2(cosine_similarity_matrix(x_embed, att_embed))
            loss.backward()
            optimizer.step()

            train_cosim_s.extend(cosim_B.detach().cpu().numpy().tolist())

        train_cosim_mean = np.mean(train_cosim_s)
        train_cosim_std = np.std(train_cosim_s)

        # --- Validation ---
        model_embed.eval()
        scores = []
        labels = []

        with torch.no_grad():
            for x_BxTxF_val, y_B_val, att_BxAxD_val in tqdm(test_loader, desc=f"Eval-{machine} Epoch {epoch + 1}"):
                x_BxTxF_val, att_BxAxD_val = x_BxTxF_val.to(device), att_BxAxD_val.to(device)

                x_embed, att_embed = model_embed(x_BxTxF_val, att_BxAxD_val)

                cosine_sim = torch.nn.functional.cosine_similarity(x_embed, att_embed, dim=1)

                score = cosine_sim.clamp(min=0.0)

                scores.append(score)
                labels.append(y_B_val)

        scores = torch.cat(scores).detach().cpu().numpy()
        labels = torch.cat(labels).detach().cpu().numpy()

        # print(scores)
        # print(labels)

        val_auc = compute_auc(labels, scores)
        scheduler.step(val_auc)

        print(f"→ Epoch {epoch + 1} | Train cosim mean ={train_cosim_mean:.6f} | Val AUC={val_auc:.6f}")

        # --- Save the best model ---
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0

            model_save_path = os.path.join(save_dir, f"EmbeddingModel.{machine}.pth")
            torch.save(model_embed.state_dict(), model_save_path)
            print(f"✅ Saved model: {model_save_path}")

            # Save evaluation results
            eval_metrics = {}
            eval_metrics['Machine'] = machine
            eval_metrics.update(compute_metrics(scores, labels))

            df_eval = pd.DataFrame([eval_metrics])
            eval_csv_path = os.path.join(save_dir, f"eval.{machine}.csv")
            df_eval.to_csv(eval_csv_path, index=False)
            print(f"✅ Saved eval metrics: {eval_csv_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🔴 Early stopping at epoch {epoch + 1}")
                break

import os
import glob
import pandas as pd

# --- Final merging ---
eval_csv_paths = glob.glob(os.path.join(save_dir, "eval.*.csv"))

dfs = []
for path in eval_csv_paths:
    if 'eval.all.csv' in path:
        continue
    df = pd.read_csv(path)
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

# --- Calculate mean row ---
mean_row = final_df.mean(numeric_only=True).round(4)
mean_df = pd.DataFrame([mean_row])

first_col_name = final_df.columns[0]
mean_df.insert(0, first_col_name, "mean")

final_df_with_mean = pd.concat([final_df, mean_df], ignore_index=True)

final_eval_path = os.path.join(save_dir, "eval.all.csv")
final_df_with_mean.to_csv(final_eval_path, index=False)

print(f"\n✅ Final merged evaluation (with mean row) saved: {final_eval_path}")
print("\n📈 Overall mean metrics:")
print(mean_row)
