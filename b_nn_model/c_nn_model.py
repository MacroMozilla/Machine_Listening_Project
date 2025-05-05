# --- Imports ---
import os

import torch
import torch.nn as nn

import preset
from f_utility.io_tools import read_json

machine2attinfos = read_json(preset.dpath_machine2attinfos) if os.path.exists(preset.dpath_machine2attinfos) else {}

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.same_shape = (in_channels == out_channels) and (stride == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels)
        )

        self.skip = (
            nn.Identity() if self.same_shape
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.skip(x))

class DualFeatureEncoder(nn.Module):
    def __init__(self, machine, raw_dim=16000, mel_dim=128, embed_dim=128):
        super().__init__()
        self.encoder_raw = RawFeatureExtractor(F_in=raw_dim, F_out=embed_dim)
        self.encoder_logmel = FeatureExtractor(F_in=mel_dim, F_out=embed_dim)
        self.embed_att = EmbedAtt(machine)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, waveform_BxT, logmel_BxTxF, att_BxA, lam):
        raw_input = waveform_BxT.unsqueeze(1)        # (B, 1, T)
        raw_feat = self.encoder_raw(raw_input)       # (B, D)

        logmel_feat = self.encoder_logmel(logmel_BxTxF)  # (B, D)

        x_embed = lam * raw_feat + (1 - lam) * logmel_feat
        x_embed = self.linear(x_embed)

        att_embed = self.embed_att(att_BxA)
        att_embed = self.linear(att_embed)

        return x_embed, att_embed

# --- Model Definitions ---
class EmbedAtt(nn.Module):
    def __init__(self, machine, out_dim=128, hidden_dim=64, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.attinfos = machine2attinfos[machine]  # List of dicts
        self.embedders = nn.ModuleList()

        for attinfo in self.attinfos:
            if attinfo['type'] in ['int', 'float']:
                # Numeric: sigmoid((x - mean) / (std + eps)) -> Linear(hidden_dim)
                embed = nn.Linear(1, hidden_dim)
            elif attinfo['type'] == 'str':
                enum_size = len(attinfo['enum']) + 1  # Add 1 for unknown/missing
                embed = nn.Embedding(enum_size, hidden_dim)
            else:
                raise ValueError(f"Unknown attribute type: {attinfo['type']}")
            self.embedders.append(embed)

        self.project_out = nn.Linear(hidden_dim * len(self.attinfos), out_dim)

    def forward(self, x):  # x: [B, A]
        B, A = x.shape
        outputs = []

        for i, attinfo in enumerate(self.attinfos):
            xi = x[:, i]  # shape [B]
            if attinfo['type'] in ['int', 'float']:
                # Normalize and sigmoid
                mean = attinfo['mean']
                std = attinfo['std']
                xi = (xi - mean) / (std + self.eps)
                xi = torch.sigmoid(xi.unsqueeze(-1))  # shape [B, 1]
                hi = self.embedders[i](xi)  # shape [B, H]
            elif attinfo['type'] == 'str':
                # Assume xi already mapped to index [0, L]
                xi = xi.long()
                hi = self.embedders[i](xi)  # shape [B, H]
            outputs.append(hi)

        h = torch.cat(outputs, dim=-1)  # shape [B, A*H]
        return self.project_out(h)  # shape [B, out_dim]



class FeatureExtractor(nn.Module):
    def __init__(self, F_in=128, F_out=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 1, T, F) → (B, 32, T, F)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=(2,2), padding=1),  # T,F → T/2,F/2
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=(2,2), padding=1),  # T,F → T/4,F/4
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.adapool = nn.AdaptiveAvgPool2d((1, 1))  # → (B, 128, 1, 1)
        self.fc = nn.Linear(128, F_out)

    def forward(self, x_TxF):  # Input: (B, T, F)
        x = x_TxF.unsqueeze(1)  # (B, 1, T, F)
        x = self.encoder(x)     # (B, 128, T//4, F//4)
        x = self.adapool(x).squeeze(-1).squeeze(-1)  # (B, 128)
        return self.fc(x)       # (B, F_out)



class RawFeatureExtractor(nn.Module):
    def __init__(self, F_in=16000, F_out=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, stride=4, padding=5),  # Downsample T → T/4
            nn.BatchNorm1d(32),
            nn.ReLU(),

            ResidualBlock1D(32, 64, stride=2),   # T → T/8
            ResidualBlock1D(64, 128, stride=2),  # T → T/16
            ResidualBlock1D(128, 128, stride=2)  # T → T/32
        )

        self.adapool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128 * 2, F_out)  # max + mean → (B, 256) → (B, 128)

    def forward(self, x):  # (B, 1, T)
        x = self.encoder(x)

        max_pooled = self.adapool(x).squeeze(-1)  # (B, 128)
        mean_pooled = x.mean(dim=-1)              # (B, 128)

        combined = torch.cat([max_pooled, mean_pooled], dim=-1)  # (B, 256)
        return self.fc(combined)  # (B, F_out)



class EmbeddingModel(nn.Module):
    def __init__(self, machine, dropout_rate=0.3, F_in=768, embed_dim=128, out_dim=128, feature_extractor=FeatureExtractor):
        super().__init__()
        self.feature_extractor = feature_extractor(F_in=F_in, F_out=embed_dim)
        self.embed_att = EmbedAtt(machine)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(embed_dim, out_dim)

    def forward(self, x_TxF, att_AxD):
        x_embed = self.feature_extractor(x_TxF)
        att_embed = self.embed_att(att_AxD)

        x_embed = self.dropout(x_embed)
        att_embed = self.dropout(att_embed)

        x_embed = self.linear(x_embed)
        att_embed = self.linear(att_embed)

        return x_embed, att_embed
