import os

import numpy as np
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

        att_types = [str(type(v).__name__) for v in self.items[0]['att']]
        cols = list(zip(*[[v for v in item['att']] for item in self.items]))

        print(f"{[part]} {[machine]}")
        for i, vtp in enumerate(att_types):
            col = cols[i]
            if vtp == 'str':

                mcats = sorted(list(set(col)))
                print(f"att-{i} {vtp} {mcats}")

            elif vtp in {'int', 'float'}:
                print(f"att-{i} {vtp} {[eval(vtp)(min(col)), eval(vtp)(max(col))]} {[round(float(np.mean(col)), 2), round(float(np.std(col)), 2)]}")
        print()
        self.att_num = len(att_types)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        item = self.items[idx]
        fpath = item['fpath']
        att = item['att']
        waveform, sample_rate = torchaudio.load(fpath)

        label = int(0.5 * (item['label'] + 1))

        waveform = waveform.squeeze(0)

        return waveform, label, att


if __name__ == '__main__':
    pass

    for machine in info[P_devtrain]:
        wd_train = WavDataset(part=P_devtrain, machine=machine)
        wd_test = WavDataset(part=P_devtest, machine=machine)

    machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']

    dataset = WavDataset(part=P_devtrain, machine='bearing')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    for batch in dataloader:
        x_BxT, y_B, atts_NxB = batch
        print(x_BxT.shape, x_BxT.dtype)
        print(y_B.shape, y_B.dtype)
        for aidx, att in enumerate(atts_NxB):
            print(f"aidx: {list(att)}")

        print()
        print()
