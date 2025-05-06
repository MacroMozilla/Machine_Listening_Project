import os

import numpy as np
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import preset
from a_prepare_data.a_prep_path import P_devtrain, P_devtest
from f_utility.io_tools import read_json, save_json

info = read_json(preset.dpath_info_json)

machine2attinfos = read_json(preset.dpath_machine2attinfos) if os.path.exists(preset.dpath_machine2attinfos) else {}


class WavDataset(Dataset):
    def __init__(self, part, machine, domain='all'):
        self.part = part
        self.machine = machine
        self.domain = domain

        self.items = []

        for item in info[part][machine]:
            if self.domain == 'all' or (self.domain == item['domain']):
                self.items.append(item)

        cols = list(zip(*[[v for v in item['att']] for item in self.items]))

        self.attinfos = []
        if os.path.exists(preset.dpath_machine2attinfos):
            self.attinfos = machine2attinfos[machine]

        else:
            for v in self.items[0]['att']:
                attinfo = {}
                attinfo['type'] = str(type(v).__name__)
                attinfo['mean'] = None
                attinfo['std'] = None
                attinfo['enum'] = None
                self.attinfos.append(attinfo)

            print(self.attinfos)
            for i, attinfo in enumerate(self.attinfos):
                vtype = attinfo['type']
                col = cols[i]
                if vtype == 'str':
                    attinfo['enum'] = sorted(list(set(col)))

                elif vtype in {'int', 'float'}:
                    vals = [eval(vtype)(val) for val in col]
                    attinfo['mean'] = float(np.mean(vals))
                    attinfo['std'] = float(np.std(vals))

        self.fctns = [lambda x: x for x in self.attinfos]

        print(f"{[part]} {[machine]}")
        for i, attinfo in enumerate(self.attinfos):
            vtype = attinfo['type']
            col = cols[i]
            if vtype == 'str':

                mcats = self.attinfos[i]['enum']
                print(f"att-{i} {vtype} {mcats}")
                self.fctns[i] = lambda x: torch.tensor(
                    mcats.index(x) if x in mcats else len(mcats)
                ).float()

            elif vtype in {'int', 'float'}:
                mean_val = self.attinfos[i]['mean']
                std_val = self.attinfos[i]['std']
                max_val = max(col)
                min_val = min(col)
                print(f"att-{i} {vtype} {[eval(vtype)(max_val), eval(vtype)(min_val)]} {[round(mean_val, 2), round(std_val, 2)]}")

                self.fctns[i] = lambda x: torch.tensor(x)

        print()

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

        processed = torch.stack(processed_list, dim=0).to(torch.float32)

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

    machine2attinfos = read_json(preset.dpath_machine2attinfos) if os.path.exists(preset.dpath_machine2attinfos) else {}
    for machine in machines:
        dataset = WavDataset(part=P_devtrain, machine=machine)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        machine2attinfos[machine] = dataset.attinfos

        for batch in dataloader:
            x_Bx1xT, y_B, atts_BxA = batch

            print(f"{x_Bx1xT.shape =}\t{x_Bx1xT.dtype=}") # [32, 1, 160000]
            print(f"{y_B.shape =}\t{x_Bx1xT.dtype=}")
            print(f"{atts_BxA.shape =}\t{x_Bx1xT.dtype=}")
            print()
            break

    print(machine2attinfos)

    save_json(machine2attinfos, preset.dpath_machine2attinfos)
