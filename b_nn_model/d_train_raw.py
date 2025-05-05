# --- Your imports ---
import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from b_nn_model.a_nn_metrics import compute_metrics
from b_nn_model.c_nn_model import DualFeatureEncoder
import preset
from a_prepare_data.a_prep_path import P_devtrain, P_devtest
from a_prepare_data.b_prep_dataset import WavDataset
from b_nn_model.a_nn_metrics import compute_auc
from b_nn_model.b_nn_loss import CosineSimilarity01, matrix_bce_loss_balanced_v2, cosine_similarity_matrix
from b_nn_model.c_nn_model import EmbeddingModel, RawFeatureExtractor  # 🛠 注意：换成了EmbeddingModel！

# Wilkinhoff
def apply_mixup(x1, x2, lam):
    return lam * x1 + (1 - lam) * x2


# --- Global settings ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 20
epochs = 50
learning_rate = 1e-3
latent_dim = 64
patience = 10
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
    model_embed = DualFeatureEncoder(machine=machine).to(device)
    optimizer = torch.optim.Adam(model_embed.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience, factor=0.5)
    cosim01 = CosineSimilarity01()

    best_auc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model_embed.train()

        train_cosim_s = []
        N = len(train_loader.dataset)
        for waveform_BxT, logmel_BxTxF, _, att_BxA in tqdm(train_loader, desc=f"Train-{machine} Epoch {epoch + 1}"):
            waveform_BxT = waveform_BxT.to(device)         # (B, T)
            logmel_BxTxF = logmel_BxTxF.to(device)         # (B, T, F)
            att_BxA = att_BxA.to(device)                   # (B, A)

            optimizer.zero_grad()

            lam = np.random.beta(0.2, 0.2)

            x_embed, att_embed = model_embed(waveform_BxT, logmel_BxTxF, att_BxA, lam)

            loss = matrix_bce_loss_balanced_v2(cosine_similarity_matrix(x_embed, att_embed))
            loss.backward()
            optimizer.step()

            cosim_B = torch.nn.functional.cosine_similarity(x_embed, att_embed, dim=1)
            train_cosim_s.extend(cosim_B.detach().cpu().numpy().tolist())


        train_cosim_mean = np.mean(train_cosim_s)
        train_cosim_std = np.std(train_cosim_s)

        # --- Validation ---
        model_embed.eval()
        scores = []
        labels = []

        with torch.no_grad():
            for waveform_val, logmel_val, y_val, att_val in tqdm(test_loader, desc=f"Eval-{machine} Epoch {epoch + 1}"):
                waveform_val = waveform_val.to(device)
                logmel_val = logmel_val.to(device)
                att_val = att_val.to(device)

                x_embed, att_embed = model_embed(waveform_val, logmel_val, att_val, lam=1.0)

                cosine_sim = torch.nn.functional.cosine_similarity(x_embed, att_embed, dim=1)
                score = cosine_sim.clamp(min=0.0)

                scores.append(score)
                labels.append(y_val)

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
