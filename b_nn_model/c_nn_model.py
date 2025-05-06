# --- Imports ---
import os

import torch
import torch.nn as nn

import preset
from f_utility.io_tools import read_json

machine2attinfos = read_json(preset.dpath_machine2attinfos) if os.path.exists(preset.dpath_machine2attinfos) else {}


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


# Original Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, F_in=768, F_out=128):
        super().__init__()
        self.F_in = F_in
        self.out_features = F_out

        # Temporal encoder: operates over time dimension
        # More flexibility
        self.encoder = nn.Sequential(
            nn.Conv1d(F_in, 256, kernel_size=5, padding=2),  # (B, 256, T)
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (B, 256, T//2)

            nn.Conv1d(256, 128, kernel_size=3, padding=1),  # (B, 128, T//2)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # (B, 128, T//4)
        )

        self.adapool = nn.AdaptiveMaxPool1d(1)  # (B, 128, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 2, F_out)  # max + mean → 256 → 128

    def forward(self, x_TxF):  # Input: (B, T, F)
        B, T, F = x_TxF.shape
        assert F == self.F_in, f"Expected F={self.F_in}, got {F}"

        x = x_TxF.transpose(1, 2)  # → (B, F, T)
        x = self.encoder(x)  # → (B, 128, T’)

        max_pooled = self.adapool(x).squeeze(-1)  # (B, 128)
        mean_pooled = x.mean(dim=-1)  # (B, 128)

        combined = torch.cat([max_pooled, mean_pooled], dim=-1)  # (B, 256)
        combined = self.dropout(combined)

        return self.fc(combined)  # (B, F_out)

# --- SE Block Definition ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # interchangeable (Squeeze)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # (Excitation)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y 

# --- Residual Block Definition ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_se=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding),
            nn.BatchNorm1d(out_channels)
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.shortcut = (nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride) \
            if in_channels != out_channels or stride > 1 else nn.Identity())

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.se(out)

        if out.shape[-1] != residual.shape[-1]:
            min_len = min(out.shape[-1], residual.shape[-1])
            out = out[:, :, :min_len]
            residual = residual[:, :, :min_len]

        return out + residual

# Feature Extractor with just SE blocks in each Convolutional Layer Block
class RawFeatureExtractorSE(nn.Module):
    def __init__(self, F_in=16000, F_out=128):
        super().__init__()
        self.out_features = F_out

        # 1 input channel for Raw Wave Input
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=11, stride=4, padding=7),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            SEBlock(8)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=7, stride=2, padding=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            SEBlock(16)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            SEBlock(32)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(32, 128, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SEBlock(128)
        )

        self.adapool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128 * 2, F_out)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        max_pooled = self.adapool(x).squeeze(-1)
        mean_pooled = x.mean(dim=-1)
        combined = torch.cat([max_pooled, mean_pooled], dim=-1)

        return self.fc(combined)

# Feature Extractor with just Residual blocks
class RawFeatureExtractorRB(nn.Module):
    def __init__(self, F_in=16000, F_out=128):
        super().__init__()
        self.out_features = F_out

        self.rb1 = ResidualBlock(768, 512, kernel_size=7, stride=2, padding=7)

        self.rb2 = ResidualBlock(512, 256, kernel_size=5, stride=2, padding=5)

        self.rb3 = ResidualBlock(256, 256, kernel_size=3, stride=2, padding=3)

        self.rb4 = ResidualBlock(256, 128, kernel_size=3, stride=2, padding=3)

        self.adapool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128 * 2, F_out)

    def forward(self, x):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)

        max_pooled = self.adapool(x).squeeze(-1)
        mean_pooled = x.mean(dim=-1)
        combined = torch.cat([max_pooled, mean_pooled], dim=-1)

        return self.fc(combined)
    
class RawFeatureExtractor(nn.Module):
    def __init__(self, F_in=16000, F_out=128):
        super().__init__()
        self.out_features = F_out

        # Efficient Conv Blocks (with downsampling via stride)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=11, stride=4, padding=5),  # (B, 32, 4000)
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Conv1d(8, 16, kernel_size=7, stride=2, padding=3),  # (B, 64, 2000)
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # (B, 128, 1000)
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 128, kernel_size=5, stride=2, padding=2),  # (B, 128, 500)
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.adapool = nn.AdaptiveMaxPool1d(1)  # (B, 128, 1)
        self.fc = nn.Linear(128 * 2, F_out)  # max + mean → (B, 256) → (B, 128)

    def forward(self, x):  # (B, 1, 16000)
        x = self.encoder(x)  # → (B, 128, 500)

        max_pooled = self.adapool(x).squeeze(-1)  # (B, 128)
        mean_pooled = x.mean(dim=-1)  # (B, 128)

        combined = torch.cat([max_pooled, mean_pooled], dim=-1)  # (B, 256)

        return self.fc(combined)  # (B, 128)


# class EmbeddingModel(nn.Module):
#     def __init__(self, machine, dropout_rate=0.3, embed_dim=128, out_dim=128, attbind='add', feature_extractor=RawFeatureExtractorRB):
#         super().__init__()
#         self.embed_wav = feature_extractor
#         self.embed_att = EmbedAtt(machine, out_dim=embed_dim, attbind=attbind)
#         self.linear_wav = nn.Linear(embed_dim, out_dim)
#         self.linear_att = nn.Linear(embed_dim, out_dim)

#         self.dropout = nn.Dropout(p=dropout_rate)

#     def forward(self, x_TxF, att_AxD):
#         x_embed = self.embed_wav(x_TxF)
#         att_embed = self.embed_att(att_AxD)

#         x_embed = self.dropout(x_embed)
#         att_embed = self.dropout(att_embed)

#         x_embed = self.linear_wav(x_embed)
#         att_embed = self.linear_att(att_embed)

#         return x_embed, att_embed
    
class EmbeddingModel(nn.Module):
    def __init__(self, machine, dropout_rate=0.25, F_in=768, embed_dim=128, out_dim=128, feature_extractor=RawFeatureExtractorRB):
        super().__init__()
        self.feature_extractor = feature_extractor(F_in=F_in, F_out=embed_dim)
        self.embed_att = EmbedAtt(machine)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(embed_dim, out_dim)

    def forward(self, x_TxF, att_AxD):

        x_TxF = x_TxF.transpose(1, 2)

        x_embed = self.feature_extractor(x_TxF)
        att_embed = self.embed_att(att_AxD)

        x_embed = self.dropout(x_embed)
        att_embed = self.dropout(att_embed)

        x_embed = self.linear(x_embed)
        att_embed = self.linear(att_embed)

        return x_embed, att_embed
