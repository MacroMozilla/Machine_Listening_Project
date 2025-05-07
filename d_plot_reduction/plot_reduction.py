# --- Your imports ---
import hashlib
import json
import os
import pathlib
from datetime import datetime
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.markers as mmarkers
import os

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
from a_prepare_data.b0_prep_dataset import WavDataset, N_machine, machine2midx
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

    # Initialize model, optimizer, etc.
    if config['datasetcls'] == 'WavDataset':
        model_embed = EmbeddingModel(dropout_rate=dropout, embed_dim=latent_dim, out_dim=output_dim, feature_extractor=RawFeatureExtractor(F_in=160000, F_out=latent_dim), embed_extractor=EmbedMachine(N_machine=N_machine, out_dim=latent_dim)).to(device)
    elif config['datasetcls'] == 'MelSpecDataset':
        model_embed = EmbeddingModel(dropout_rate=dropout, embed_dim=latent_dim, out_dim=output_dim, feature_extractor=FeatureExtractor(F_in=128, F_out=latent_dim), embed_extractor=EmbedMachine(N_machine=N_machine, out_dim=latent_dim)).to(device)
    elif config['datasetcls'] == 'Wav2VecDataset':
        model_embed = EmbeddingModel(dropout_rate=dropout, embed_dim=latent_dim, out_dim=output_dim, feature_extractor=FeatureExtractor(F_in=768, F_out=latent_dim), embed_extractor=EmbedMachine(N_machine=N_machine, out_dim=latent_dim)).to(device)
    else:
        raise Exception(f"{config['datasetcls']=} is not supported!")

    fpath = config.get('fpath', None)

    model_embed.load_state_dict(torch.load(fpath))

    model_embed.eval()
    with torch.no_grad():
        items = []
        sidx = 0
        for machine in machines:
            # Prepare datasets
            train_dataset = datasetcls(part=P_devtrain, machine=machine)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            test_dataset_all = datasetcls(part=P_devtest, machine=machine, domain='all')
            test_loader_all = DataLoader(test_dataset_all, batch_size=batch_size, shuffle=False, num_workers=0)

            for name, dataloader in zip(['train-all', 'test-all'], [train_loader, test_loader_all]):

                embeds = []
                y_B_s = []
                for x_BxTxF_val, y_B_val, _, machine_B_val in tqdm(dataloader, desc=f"Eval-{machine}-{name}"):
                    x_BxTxF_val = x_BxTxF_val.to(device)

                    embed_BxF = torch.nn.functional.normalize(model_embed.linear_wav(model_embed.embed_wav(x_BxTxF_val)), p=2, dim=1)
                    embeds.append(embed_BxF)
                    y_B_s.append(y_B_val)

                data = torch.concat(embeds)
                y = torch.concat(y_B_s)

                if name == 'train-all':
                    item = {}
                    item['machine'] = machine

                    item['name'] = f"{name}-1"
                    cur_data = data
                    item['data'] = cur_data
                    item['sidx'] = sidx
                    item['eidx'] = sidx + cur_data.shape[0]
                    sidx = item['eidx']
                    items.append(item)
                else:
                    item = {}
                    item['machine'] = machine
                    item['name'] = f"{name}-1"
                    cur_data = data[y >= 0.5]
                    item['data'] = cur_data
                    item['sidx'] = sidx
                    item['eidx'] = sidx + cur_data.shape[0]
                    sidx = item['eidx']
                    items.append(item)

                    item = {}
                    item['machine'] = machine
                    item['name'] = f"{name}-0"
                    cur_data = data[y < 0.5]
                    item['data'] = cur_data
                    item['sidx'] = sidx
                    item['eidx'] = sidx + cur_data.shape[0]
                    sidx = item['eidx']
                    items.append(item)

        for machine in machines:
            item = {}
            item['machine'] = machine
            item['name'] = 'machine-category'
            cur_data = torch.nn.functional.normalize(model_embed.linear_att(model_embed.embed_att(torch.tensor([machine2midx[machine]]).to(device, dtype=torch.int64))), p=2, dim=1)
            item['data'] = cur_data
            item['sidx'] = sidx
            item['eidx'] = sidx + cur_data.shape[0]
            sidx = item['eidx']
            items.append(item)

    # Combine all data
    all_data = torch.cat([item['data'] for item in items], dim=0)
    data_array = all_data.cpu().numpy()

    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(data_array)

    # 样式设置
    machine_set = sorted(set(item['machine'] for item in items))

    # 每个 machine 一个颜色
    color_map = {machine: cm.tab10(i) for i, machine in enumerate(machine_set)}

    # 每个 name 一个样式（含大小）
    name_styles = {
        'train-all-1': {
            'marker': 's',
            'facecolors': 'none',
            'alpha': 0.25,
            'size': 40
        },
        'test-all-1': {
            'marker': 'o',
            'facecolors': 'none',
            'alpha': 0.25,
            'size': 40
        },
        'test-all-0': {
            'marker': 'X',
            'facecolors': 'none',
            'alpha': 0.25,
            'size': 40
        },
        'machine-category': {
            'marker': '+',
            'facecolors': 'full',
            'alpha': 1.0,
            'size': 40 * 25
        }
    }

    plt.figure(figsize=(20, 16))
    plotted_labels = set()
    legend_handles = []
    for item in items:
        s, e = item['sidx'], item['eidx']
        color = color_map[item['machine']]
        style = name_styles[item['name']]

        label = f"{item['machine']}_{item['name']}"
        show_label = label not in plotted_labels

        plt.scatter(
            reduced[s:e, 0], reduced[s:e, 1],
            color=color,
            marker=style['marker'],
            edgecolors=color,
            alpha=style['alpha'],
            s=style['size'],
            label=None
        )

        if show_label:
            handle = plt.scatter(
                [], [],
                color=color,
                marker=style['marker'],
                edgecolors=color,
                alpha=1.0,
                s=40,
                label=label
            )
            legend_handles.append(handle)
            plotted_labels.add(label)

    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"[{config['datasetcls']}] t-SNE projection by machine", fontsize=24)
    plt.tight_layout()

    fpath = pathlib.Path(fpath)
    second_level_dir = fpath.parent.name
    output_path = os.path.join(preset.dpath_custom_models, f"{second_level_dir}.png")

    print(second_level_dir)

    plt.savefig(output_path)
    plt.close()


if __name__ == '__main__':
    pass

    config = {
        'batch_size': 25,
        'epochs': 50,
        'learning_rate': 1e-3,
        'latent_dim': 256,
        'output_dim': 256,
        'patience': 15,
        'dropout': 0.125,
        'datasetcls': 'WavDataset',
        'fpath': r'D:\b_data_train\data_a_raw\DCASE_2023_task2\custom_models\nonatt_WavDataset_6936f0f514_250507_134911\EmbeddingModel.all.pth',
        'lossfctncls': 'CosineSimilarityLoss',
        'attbind': 'add',
    }
    pprint(config)

    train_with_config_nonatt(config)

    config = {
        'batch_size': 25,
        'epochs': 50,
        'learning_rate': 1e-3,
        'latent_dim': 256,
        'output_dim': 256,
        'patience': 15,
        'dropout': 0.125,
        'datasetcls': 'MelSpecDataset',
        'fpath': r'D:\b_data_train\data_a_raw\DCASE_2023_task2\custom_models\nonatt_MelSpecDataset_f37b0d8f09_250507_142915\EmbeddingModel.all.pth',
        'lossfctncls': 'CosineSimilarityLoss',
        'attbind': 'add',
    }
    pprint(config)

    train_with_config_nonatt(config)

    config = {
        'batch_size': 25,
        'epochs': 50,
        'learning_rate': 1e-3,
        'latent_dim': 256,
        'output_dim': 256,
        'patience': 15,
        'dropout': 0.125,
        'datasetcls': 'Wav2VecDataset',
        'fpath': r'D:\b_data_train\data_a_raw\DCASE_2023_task2\custom_models\nonatt_Wav2VecDataset_208e118112_250507_141438\EmbeddingModel.all.pth',
        'lossfctncls': 'CosineSimilarityLoss',
        'attbind': 'add',
    }
    pprint(config)

    train_with_config_nonatt(config)
