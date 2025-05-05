import os
import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

import preset
from a_prepare_data.a_prep_path import P_devtrain, P_devtest
from f_utility.io_tools import read_json

info = read_json(preset.dpath_info_json)
machine2attinfos = read_json(preset.dpath_machine2attinfos) if os.path.exists(preset.dpath_machine2attinfos) else {}

class WavDataset(Dataset):
    def __init__(self, part, machine):
        self.part = part
        self.machine = machine
        self.items = info[part][machine]

        cols = list(zip(*[[v for v in item['att']] for item in self.items]))
        self.attinfos = machine2attinfos.get(machine, [])
        
        if not self.attinfos:
            self.attinfos = []
            for v in self.items[0]['att']:
                self.attinfos.append({'type': str(type(v).__name__), 'mean': None, 'std': None, 'enum': None})

            for i, attinfo in enumerate(self.attinfos):
                vtype = attinfo['type']
                col = cols[i]
                if vtype == 'str':
                    attinfo['enum'] = sorted(list(set(col)))
                elif vtype in {'int', 'float'}:
                    vals = [eval(vtype)(val) for val in col]
                    attinfo['mean'] = float(np.mean(vals))
                    attinfo['std'] = float(np.std(vals))

        self.fctns = []
        for i, attinfo in enumerate(self.attinfos):
            vtype = attinfo['type']
            if vtype == 'str':
                mcats = attinfo['enum']
                self.fctns.append(lambda x, mcats=mcats: torch.tensor(mcats.index(x) if x in mcats else len(mcats)).float())
            else:
                self.fctns.append(lambda x: torch.tensor(x))

        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            win_length=2048
        )

    def get_label(self, idx):
        
        label = int(0.5 * (self.items[idx]['label'] + 1))
        return label

    def get_waveform(self, idx):
        fpath = self.items[idx]['fpath']
        try:
            waveform, sr = torchaudio.load(fpath)
        except:
            waveform, sr = librosa.load(fpath, sr=None, mono=True)
            waveform = torch.tensor(waveform).unsqueeze(0)
        return waveform  # shape: (1, T)

    def get_att(self, idx):
        att = self.items[idx]['att']
        return torch.stack([self.fctns[i](att[i]) for i in range(len(att))]).float()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        waveform = self.get_waveform(idx).squeeze(0)            # (T,)
        logmel = self.mel_transform(waveform.unsqueeze(0))      # (1, n_mels, time)
        logmel = torch.log(logmel + 1e-6).squeeze(0).transpose(0, 1)  # (time, n_mels)
        label = self.get_label(idx)
        att = self.get_att(idx)
        return waveform, logmel, label, att
