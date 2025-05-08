import os

import numpy as np
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import preset
from a_prepare_data.a_prep_path import P_devtrain, P_devtest
from f_utility.io_tools import read_json, save_json

info = read_json(preset.dpath_info_json)

machine2attinfos = read_json(preset.dpath_machine2attinfos) if os.path.exists(preset.dpath_machine2attinfos) else {}

machine2midx = {machine: midx for midx, machine in enumerate(sorted(list(machine2attinfos.keys())))}

N_machine = len(machine2attinfos)


class WavDataset(Dataset):
    def __init__(self, part, machine, domain='all'):
        self.part = part
        self.machine = machine
        self.domain = domain

        self.items = []

        if self.machine == 'all':
            for machine, items in info[part].items():
                for item in items:
                    if self.domain == 'all' or (self.domain == item['domain']):
                        item['machine'] = machine
                        self.items.append(item)
        else:
            for item in info[part][machine]:
                if self.domain == 'all' or (self.domain == item['domain']):
                    item['machine'] = machine
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

                # print(self.attinfos)
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

            # print(f"{[part]} {[machine]}")
            for i, attinfo in enumerate(self.attinfos):
                vtype = attinfo['type']
                col = cols[i]
                if vtype == 'str':

                    mcats = self.attinfos[i]['enum']
                    # print(f"att-{i} {vtype} {mcats}")
                    self.fctns[i] = lambda x: torch.tensor(
                        mcats.index(x) if x in mcats else len(mcats)
                    ).float()

                elif vtype in {'int', 'float'}:
                    mean_val = self.attinfos[i]['mean']
                    std_val = self.attinfos[i]['std']
                    max_val = max(col)
                    min_val = min(col)
                    # print(f"att-{i} {vtype} {[eval(vtype)(max_val), eval(vtype)(min_val)]} {[round(mean_val, 2), round(std_val, 2)]}")

                    self.fctns[i] = lambda x: torch.tensor(x)

            # print()

    def get_label(self, idx):
        item = self.items[idx]
        label = int(0.5 * (item['label'] + 1))
        return label

    def get_waveform(self, idx):
        item = self.items[idx]
        fpath = item['fpath']
        waveform, sample_rate = torchaudio.load(fpath)
        target_len = 192000
        waveform = torch.nn.functional.pad(waveform[:, :target_len], (0, max(0, target_len - waveform.shape[1])))
        return waveform

    def get_machine(self, idx):
        item = self.items[idx]
        machine = item['machine']
        return torch.tensor(machine2midx[machine]).to(dtype=torch.int64)

    def get_att(self, idx):

        if self.machine == 'all':
            return torch.tensor([])

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

        machine = self.get_machine(idx)

        return waveform, label, att, machine


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
            x_Bx1xT, y_B, atts_BxA, machine_B = batch

            print(f"{x_Bx1xT.shape =}\t{x_Bx1xT.dtype=}")  # [32, 1, 160000]
            print(f"{y_B.shape =}\t{y_B.dtype=}")
            print(f"{atts_BxA.shape =}\t{atts_BxA.dtype=}")
            print(f"{machine_B.shape =}\t{machine_B.dtype=}")
            print()
            break

    print(machine2attinfos)

    save_json(machine2attinfos, preset.dpath_machine2attinfos)
