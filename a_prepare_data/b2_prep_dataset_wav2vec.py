import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model

from a_prepare_data.a_prep_path import P_devtrain, P_devtest
from a_prepare_data.b0_prep_dataset import WavDataset

# --- Device setting ---
device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load pretrained Wav2Vec2 model ---
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model = model.to(device_gpu)
model.eval()


class Wav2VecDataset(Dataset):

    def __init__(self, part, machine, domain='all'):
        self.wavdataset = WavDataset(part, machine, domain=domain)

    def __len__(self):
        return len(self.wavdataset)

    def __getitem__(self, idx):
        item = self.wavdataset.items[idx]
        fpath = item['fpath']

        fpath_cache = fpath.replace('.wav', '.pth')

        if os.path.exists(fpath_cache):
            x_TxF = torch.load(fpath_cache)
        else:
            waveform = self.wavdataset.get_waveform(idx)  # Load waveform: [1, T]
            waveform = waveform.to(device_gpu).float()

            with torch.no_grad():
                outputs = model(waveform)
                x_TxF = outputs.last_hidden_state.squeeze(0).cpu()  # [T, F]

            torch.save(x_TxF, fpath_cache)

        target_T = 599
        T, F = x_TxF.shape
        x_TxF = torch.nn.functional.pad(x_TxF[:target_T], (0, 0, 0, max(0, target_T - T)))  # Pad on T

        label = self.wavdataset.get_label(idx)
        att = self.wavdataset.get_att(idx)
        machine = self.wavdataset.get_machine(idx)
        return x_TxF, label, att,machine


if __name__ == '__main__':
    parts = [P_devtrain, P_devtest]
    machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']

    # --- Loop over all parts and machines ---
    for part in parts:
        for machine in machines:
            dataset = Wav2VecDataset(part=part, machine=machine)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            print(f"Building cache for part: {part}, machine: {machine}, total: {len(dataset)} samples")

            for batch in tqdm(dataloader):
                x_TxF, y_B, att,machine_B = batch
                print(x_TxF.shape)

                pass
