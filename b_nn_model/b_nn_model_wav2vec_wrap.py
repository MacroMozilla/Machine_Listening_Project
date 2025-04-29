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
    def __init__(self, F_in=768, out_features=128):
        super().__init__()
        self.F_in = F_in
        self.out_features = out_features
        self.pool_T = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(self.F_in * 3, self.out_features)  # 注意是 *3

    def forward(self, x_TxF):
        B, T, F = x_TxF.shape
        assert F == self.F_in, f"Expected F={self.F_in}, but got F={F}"

        x = x_TxF.transpose(1, 2)   # (B, F, T)

        # Max Pool
        max_pooled = self.pool_T(x).squeeze(-1)  # (B, F)

        # Min Pool
        min_pooled = -self.pool_T(-x).squeeze(-1)  # (B, F)

        # Mean Pool
        mean_pooled = x.mean(dim=-1)               # (B, F)

        # Concatenate
        combined = torch.cat([max_pooled, min_pooled, mean_pooled], dim=-1)  # (B, F*3)

        # Final Linear
        output = self.fc(combined)  # (B, out_features)

        return output

class EmbeddingModel(nn.Module):
    def __init__(self, machine, dropout_rate=0.3, embed_dim=128, out_dim=128):
        super().__init__()
        self.feature_extractor = FeatureExtractor()  # 输出embed_dim
        self.embed_att = EmbedAtt(machine)           # 输出embed_dim
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(embed_dim, out_dim)  # 新加的线性层

    def forward(self, x_TxF, att_AxD):
        x_embed = self.feature_extractor(x_TxF)
        att_embed = self.embed_att(att_AxD)

        x_embed = self.dropout(x_embed)
        att_embed = self.dropout(att_embed)

        x_embed = self.linear(x_embed)
        att_embed = self.linear(att_embed)

        return x_embed, att_embed