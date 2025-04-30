# --- Imports ---
import torch
import torch.nn as nn

machine2AxD = {
    'bearing': (2, 9), 'fan': (1, 5), 'gearbox': (2, 1),
    'slider': (2, 1), 'ToyCar': (3, 6), 'ToyTrain': (3, 11), 'valve': (1, 1)
}


# --- Model Definitions ---
class EmbedAtt(nn.Module):
    def __init__(self, machine, out_dim=128, hidden_dim=64):
        super().__init__()
        self.A, self.D = machine2AxD[machine]

        # Shared MLP encoder for each attribute vector
        self.encoder = nn.Sequential(
            nn.Linear(self.D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Final projection layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.ReLU()
        )

    def forward(self, att_AxD):  # shape: (B, A, D)
        B, A, D = att_AxD.shape

        x = self.encoder(att_AxD)  # → (B, A, H)

        max_pooled = x.max(dim=1).values  # (B, H)
        mean_pooled = x.mean(dim=1)  # (B, H)

        combined = torch.cat([max_pooled, mean_pooled], dim=-1)  # (B, 2H)

        return self.fc(combined)  # (B, 128)


class FeatureExtractor(nn.Module):
    def __init__(self, F_in=768, F_out=128):
        super().__init__()
        self.F_in = F_in
        self.out_features = F_out

        # Temporal encoder: operates over time dimension
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
