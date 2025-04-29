import os

import numpy as np
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import preset
from a_prepare_data.a_prep_path import P_devtrain, P_devtest
from f_utility.io_tools import read_json

info = read_json(preset.dpath_info_json)


class WavDataset(Dataset):
    def __init__(self, part, machine):
        self.part = part
        self.machine = machine

        self.items = info[part][machine]

        self.att_types = [str(type(v).__name__) for v in self.items[0]['att']]
        # self.weights = [1 for _ in self.att_types]
        self.fctns = [lambda x:x for x in self.att_types]

        cols = list(zip(*[[v for v in item['att']] for item in self.items]))

        print(f"{[part]} {[machine]}")
        for i, vtp in enumerate(self.att_types):
            col = cols[i]
            if vtp == 'str':

                mcats = sorted(list(set(col)))
                print(f"att-{i} {vtp} {mcats}")

                self.fctns[i] = lambda x: torch.nn.functional.one_hot(
                    torch.tensor(
                        mcats.index(x) if x in mcats else len(mcats)
                    ),
                    num_classes=len(mcats) + 1
                ).float()

            elif vtp in {'int', 'float'}:
                mean = float(np.mean(col))
                std = float(np.std(col))
                max_val = max(col)
                min_val = min(col)
                print(f"att-{i} {vtp} {[eval(vtp)(max_val), eval(vtp)(min_val)]} {[round(mean, 2), round(std, 2)]}")

                self.fctns[i] = lambda x: torch.sigmoid(
                    torch.tensor([(x - mean) / (std + 1e-8)])
                )

        print()
        self.att_num = len(self.att_types)

    def get_label(self, idx):
        item = self.items[idx]
        label = int(0.5 * (item['label'] + 1))
        return label

    def get_waveform(self, idx):
        item = self.items[idx]
        fpath = item['fpath']
        waveform, sample_rate = torchaudio.load(fpath)
        return waveform

    def get_att(self, idx):
        item = self.items[idx]
        att = item['att']

        processed_list = [
            self.fctns[i](att[i])
            for i in range(len(att))
        ]

        max_len = max(p.shape[0] for p in processed_list)

        padded_list = []
        for p in processed_list:
            if p.shape[0] < max_len:
                repeat_times = (max_len + p.shape[0] - 1) // p.shape[0]  # 向上取整
                p = p.repeat(repeat_times, *[1] * (p.dim() - 1))[:max_len]
            padded_list.append(p)

        processed = torch.stack(padded_list, dim=0).to(torch.float32)

        return processed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        att = self.get_att(idx)
        label = self.get_label(idx)
        waveform = self.get_waveform(idx)

        return waveform, label, att





if __name__ == '__main__':
    pass

    for machine in info[P_devtrain]:
        wd_train = WavDataset(part=P_devtrain, machine=machine)
        wd_test = WavDataset(part=P_devtest, machine=machine)

    machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']

    machine2AxD = {}

    for machine in machines:
        dataset = WavDataset(part=P_devtrain, machine=machine)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

        for batch in dataloader:
            x_BxT, y_B, atts_BxAxD = batch
            A = atts_BxAxD.shape[1]
            D = atts_BxAxD.shape[2]
            print(atts_BxAxD)
            machine2AxD[machine] = (A, D)
            break

    print(machine2AxD)