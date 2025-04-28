# --- Imports ---
import torch
import torch.nn as nn

machine2AxD = {
    'bearing': (2, 9), 'fan': (1, 5), 'gearbox': (2, 1),
    'slider': (2, 1), 'ToyCar': (3, 6), 'ToyTrain': (3, 11), 'valve': (1, 1)
}

# --- Model Definitions ---
class EmbedAtt(nn.Module):
    def __init__(self, machine):
        super().__init__()
        self.A, self.D = machine2AxD[machine]
        self.proj = nn.Linear(self.D, 16)
        self.out = nn.Linear(self.A * 16, 128)

    def forward(self, att_AxD):
        x = self.proj(att_AxD)   # shape: (B, A, 16)
        x = x.view(x.size(0), -1)  # flatten
        return self.out(x)         # (B, 128)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(128, 128)

    def forward(self, x_TxF):
        B, T, F = x_TxF.shape
        x = x_TxF.unsqueeze(1)      # (B, 1, T, F)
        x = self.conv(x)            # (B, 128, T, F)
        x = self.pool(x)            # (B, 128, 1, 1)
        x = x.view(B, 128)           # (B, 128)
        return self.fc(x)            # (B, 128)

class EmbeddingModel(nn.Module):
    def __init__(self, machine, dropout_rate=0.3):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.embed_att = EmbedAtt(machine)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_TxF, att_AxD):
        x_embed = self.feature_extractor(x_TxF)   # (B, 128)
        att_embed = self.embed_att(att_AxD)        # (B, 128)
        x_embed = self.dropout(x_embed)
        att_embed = self.dropout(att_embed)
        return x_embed, att_embed
