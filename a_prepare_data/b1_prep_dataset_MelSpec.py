import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model

from a_prepare_data.a_prep_path import P_devtrain, P_devtest
from a_prepare_data.b0_prep_dataset import WavDataset

# --- Device setting ---
device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MelSpecDataset(Dataset):
    def __init__(self, part, machine, domain='all'):
        self.wavdataset = WavDataset(part, machine, domain=domain)

        # Internal config (not exposed)
        self._sample_rate = 16000
        self._n_mels = 128
        self._win_length = 1024
        self._hop_length = 512

        # Define transforms using torchaudio module directly
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate,
            n_fft=self._win_length,
            win_length=self._win_length,
            hop_length=self._hop_length,
            n_mels=self._n_mels,
            window_fn=torch.hann_window,
            power=2.0,
        ).to(device_gpu)

        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power').to(device_gpu)

    def __len__(self):
        return len(self.wavdataset)

    def __getitem__(self, idx):
        item = self.wavdataset.items[idx]
        fpath = item['fpath']
        fpath_cache = fpath.replace('.wav', '.melspec.pth')

        if os.path.exists(fpath_cache):
            x_TxF = torch.load(fpath_cache)
        else:
            waveform = self.wavdataset.get_waveform(idx).to(device_gpu).float()  # [1, T]

            with torch.no_grad():
                melspec = self.mel_transform(waveform)  # [1, F, T]
                melspec_db = self.db_transform(melspec)  # [1, F, T]
                x_TxF = melspec_db.squeeze(0).transpose(0, 1).cpu()  # [T, F]

            torch.save(x_TxF, fpath_cache)

        target_T = 376
        T, F = x_TxF.shape
        x_TxF = torch.nn.functional.pad(x_TxF[:target_T], (0, 0, 0, max(0, target_T - T)))  # Pad on T


        label = self.wavdataset.get_label(idx)
        att = self.wavdataset.get_att(idx)
        machine = self.wavdataset.get_machine(idx)
        return x_TxF, label, att, machine


if __name__ == '__main__':
    parts = [P_devtrain, P_devtest]
    machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']

    # --- Loop over all parts and machines ---
    for part in parts:
        for machine in machines:
            dataset = MelSpecDataset(part=part, machine=machine)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            print(f"Building cache for part: {part}, machine: {machine}, total: {len(dataset)} samples")

            for batch in tqdm(dataloader):
                x_TxF, y_B, att, machine_B = batch
                # print(x_TxF.shape)
                # break
