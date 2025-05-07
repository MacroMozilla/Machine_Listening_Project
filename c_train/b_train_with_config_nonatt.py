# --- Your imports ---
import hashlib
import json
import os
from datetime import datetime
from pprint import pprint

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from a_prepare_data.b1_prep_dataset_MelSpec import MelSpecDataset
from a_prepare_data.b2_prep_dataset_wav2vec import Wav2VecDataset
from b_nn_model.a_nn_metrics import compute_metrics
import preset
from a_prepare_data.a_prep_path import P_devtrain, P_devtest
from a_prepare_data.b0_prep_dataset import WavDataset, N_machine
from b_nn_model.a_nn_metrics import compute_auc
from b_nn_model.b_nn_loss import CosineSimilarity01, matrix_bce_loss_balanced_v2, cosine_similarity_matrix, LearnableAdaCosLoss
from b_nn_model.c_nn_model import EmbeddingModel, RawFeatureExtractor, FeatureExtractor, EmbedAtt, EmbedMachine  # 🛠 注意：换成了EmbeddingModel！

import os
import glob
import pandas as pd

from f_utility.io_tools import save_json, save_jsonl, read_jsonl
from f_utility.watch import datetime2fbody

# --- Global settings ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def hash_config_to_hex(config) -> str:
    """
    将任意 Python 配置对象（例如 list of dict）hash 成 hex 字符串。
    """
    # 将 config 序列化为 JSON 字符串（排序保证一致性）
    config_str = json.dumps(config, sort_keys=True)
    # 生成 SHA256 哈希
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    return hash_obj.hexdigest()


def train_with_config_nonatt(config):
    batch_size = config.get('batch_size', 20)
    epochs = config.get('epochs', 50)
    learning_rate = config.get('learning_rate', 1e-3)
    latent_dim = config.get('latent_dim', 128)
    output_dim = config.get('output_dim', 128)
    patience = config.get('patience', 15)
    dropout = config.get('dropout', 0.25)
    attbind = config.get('attbind', 'add')

    datasetcls = {'WavDataset': WavDataset, 'Wav2VecDataset': Wav2VecDataset, 'MelSpecDataset': MelSpecDataset}.get(config['datasetcls'])

    lossfctncls = {'CosineSimilarityLoss': CosineSimilarity01, 'LearnableAdaCosLoss': LearnableAdaCosLoss}.get(config['lossfctncls'])

    fname = f"nonatt_{datasetcls.__name__}_{hash_config_to_hex(config)[:10]}_{datetime2fbody(datetime.now())}"
    save_dir = os.path.join(preset.dpath_custom_models, fname)
    os.makedirs(save_dir, exist_ok=True)

    machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve'][:]
    all_eval_results = []

    writer = SummaryWriter(save_dir)

    # Prepare datasets
    train_dataset = datasetcls(part=P_devtrain, machine='all')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize model, optimizer, etc.
    if config['datasetcls'] == 'WavDataset':
        model_embed = EmbeddingModel(dropout_rate=dropout, embed_dim=latent_dim, out_dim=output_dim, feature_extractor=RawFeatureExtractor(F_in=160000, F_out=latent_dim), embed_extractor=EmbedMachine(N_machine=N_machine, out_dim=latent_dim)).to(device)
    elif config['datasetcls'] == 'MelSpecDataset':
        model_embed = EmbeddingModel(dropout_rate=dropout, embed_dim=latent_dim, out_dim=output_dim, feature_extractor=FeatureExtractor(F_in=128, F_out=latent_dim), embed_extractor=EmbedMachine(N_machine=N_machine, out_dim=latent_dim)).to(device)
    elif config['datasetcls'] == 'Wav2VecDataset':
        model_embed = EmbeddingModel(dropout_rate=dropout, embed_dim=latent_dim, out_dim=output_dim, feature_extractor=FeatureExtractor(F_in=768, F_out=latent_dim), embed_extractor=EmbedMachine(N_machine=N_machine, out_dim=latent_dim)).to(device)
    else:
        raise Exception(f"{config['datasetcls']=} is not supported!")

    optimizer = torch.optim.AdamW(model_embed.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    lossfctn = lossfctncls()

    best_auc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model_embed.train()

        train_cosim_s = []
        N = len(train_loader.dataset)
        for x_BxTxF, _, _, machine_B in tqdm(train_loader, desc=f"Train-all Epoch {epoch + 1}"):
            x_BxTxF, machine_B = x_BxTxF.to(device), machine_B.to(device)

            B, T, F = x_BxTxF.shape

            optimizer.zero_grad()
            x_embed, machine_embed = model_embed(x_BxTxF, machine_B)

            if config['lossfctncls'] == 'CosineSimilarityLoss':
                cosim_B = lossfctn(x_embed, machine_embed)
                loss = matrix_bce_loss_balanced_v2(cosine_similarity_matrix(x_embed, machine_embed))
            elif config['lossfctncls'] == 'LearnableAdaCosLoss':
                cosim_B = lossfctn(x_embed, machine_embed)
                loss = cosim_B.mean()

            loss.backward()
            optimizer.step()

            train_cosim_s.extend(cosim_B.clamp(min=0.0).detach().cpu().numpy().tolist())

        train_cosim_mean = np.mean(train_cosim_s)

        test_auc_s = []
        test_source_auc_s = []
        test_target_auc_s = []

        machine2domain2scores = {}
        machine2domain2labels = {}
        # --- Main loop ---
        for machine in machines:
            print(f"\n===== Training Machine: {machine} =====")

            test_dataset_source = datasetcls(part=P_devtest, machine=machine, domain='source')
            test_loader_source = DataLoader(test_dataset_source, batch_size=batch_size, shuffle=False, num_workers=0)

            test_dataset_target = datasetcls(part=P_devtest, machine=machine, domain='target')
            test_loader_target = DataLoader(test_dataset_target, batch_size=batch_size, shuffle=False, num_workers=0)

            # --- Validation ---
            model_embed.eval()

            domain2scores = {}
            domain2labels = {}
            for domain, dataloader in zip(['source', 'target'], [test_loader_source, test_loader_target]):
                scores = []
                labels = []

                with torch.no_grad():
                    for x_BxTxF_val, y_B_val, _, machine_B_val in tqdm(dataloader, desc=f"Eval-{machine} Epoch {epoch + 1}"):
                        x_BxTxF_val, machine_B_val = x_BxTxF_val.to(device), machine_B_val.to(device)

                        x_embed, machine_embed = model_embed(x_BxTxF_val, machine_B_val)

                        cosine_sim = torch.nn.functional.cosine_similarity(x_embed, machine_embed, dim=1)

                        score = cosine_sim.clamp(min=0.0)

                        scores.append(score)
                        labels.append(y_B_val)

                scores = torch.cat(scores).detach().cpu().numpy()
                labels = torch.cat(labels).detach().cpu().numpy()
                domain2scores[domain] = scores
                domain2labels[domain] = labels
            machine2domain2scores[machine] = domain2scores
            machine2domain2labels[machine] = domain2labels

            # print(scores)
            # print(labels)

            test_auc_source = compute_auc(domain2labels['source'], domain2scores['source'])
            test_auc_target = compute_auc(domain2labels['target'], domain2scores['target'])
            test_auc = (test_auc_source + test_auc_target) / 2
            test_auc_s.append(test_auc)
            test_source_auc_s.append(test_auc_source)
            test_target_auc_s.append(test_auc_target)

            print(f"→ Epoch {epoch + 1} | [{machine}] | Train cosim mean = {train_cosim_mean:.6f} | Test (source,target) AUC = {test_auc_source:.6f},{test_auc_target:.6f} = {test_auc:.6f}")

        test_auc = np.mean(test_auc_s)
        test_source_auc = np.mean(test_source_auc_s)
        test_target_auc = np.mean(test_target_auc_s)

        writer.add_scalar(f'all/train/all/cosim', train_cosim_mean, epoch)
        writer.add_scalar(f'all/test/all/auc', test_auc, epoch)

        print(f"→ Epoch {epoch + 1} | [all] | Train cosim mean = {train_cosim_mean:.6f} | Test (source,target) AUC = {test_source_auc:.6f},{test_target_auc:.6f} = {test_auc:.6f}")

        scheduler.step(test_auc)
        # --- Save the best model ---
        if not np.isnan(test_auc) and test_auc > best_auc:
            best_auc = test_auc
            patience_counter = 0

            model_save_path = os.path.join(save_dir, f"EmbeddingModel.all.pth")
            torch.save(model_embed.state_dict(), model_save_path)
            print(f"✅ Saved model: {model_save_path}")

            for machine in machines:

                # Save evaluation results

                eval_infos = []
                for domain in ['source', 'target']:
                    scores = machine2domain2scores[machine][domain]
                    labels = machine2domain2labels[machine][domain]

                    eval_info = {}
                    eval_info['domain'] = domain
                    eval_info['Machine'] = machine
                    eval_info.update(compute_metrics(scores, labels))
                    eval_infos.append(eval_info)

                eval_csv_path = os.path.join(save_dir, f"eval.{machine}.jsonl")
                save_jsonl(eval_infos, eval_csv_path)

                print(f"✅ Saved eval metrics: {eval_csv_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🔴 Early stopping at epoch {epoch + 1}")
                break

    writer.close()
    # --- Final merging ---
    eval_csv_paths = glob.glob(os.path.join(save_dir, "eval.*.jsonl"))

    evalinfos = []
    for path in eval_csv_paths:
        if 'eval.mean.jsonl' in path:
            continue
        evalinfos.extend(read_jsonl(path))

    evalinfos_mean = []
    for domain in ['source', 'target', 'all']:

        evalinfos_domain = [einfo for einfo in evalinfos if (einfo['domain'] == domain) or (domain == 'all')]

        evalinfo_mean = {}
        evalinfo_mean['domain'] = domain
        for key in ['AUC', 'pAUC', 'F1', 'Accuracy', 'Precision', 'Recall']:
            val = float(np.mean([einfo[key] for einfo in evalinfos_domain]).round(4))
            evalinfo_mean[key] = val
        evalinfos_mean.append(evalinfo_mean)

    final_eval_path = os.path.join(save_dir, "eval.mean.jsonl")
    save_jsonl(evalinfos_mean, final_eval_path)

    print(f"\n✅ Final merged evaluation (with mean row) saved: {final_eval_path}")
    print("\n📈 Overall mean metrics:")

    config['mAUC'] = [einfo for einfo in evalinfos_mean if einfo['domain'] == 'all'][0]['AUC']
    config['fname'] = fname

    save_json(config, os.path.join(save_dir, "config.json"))

    save_jsonl([config], os.path.join(preset.dpath_custom_models, "smry.jsonl"), mode='a')

    infos = read_jsonl(os.path.join(preset.dpath_custom_models, "smry.jsonl"))
    infos.sort(key=lambda x: x['mAUC'], reverse=True)
    df = pd.DataFrame(infos)
    df.to_csv(os.path.join(preset.dpath_custom_models, "smry.csv"), index=False)


if __name__ == '__main__':
    pass

    dims = [256, 512, 1024, 2048]

    for datasetcls in ['WavDataset', 'MelSpecDataset', 'Wav2VecDataset'][:]:
        for attbind in ['add']:
            for lossfctncls in ['CosineSimilarityLoss', 'LearnableAdaCosLoss'][:1]:
                for i, latent_dim in enumerate([256]):
                    for j, output_dim in enumerate([256]):
                        for dropout in [0.125]:
                            config = {
                                'batch_size': 25,
                                'epochs': 50,
                                'learning_rate': 1e-3,
                                'latent_dim': latent_dim,
                                'output_dim': output_dim,
                                'patience': 15,
                                'dropout': dropout,
                                'datasetcls': datasetcls,
                                'lossfctncls': lossfctncls,
                                'attbind': attbind,
                            }
                            pprint(config)
                            train_with_config_nonatt(config)
