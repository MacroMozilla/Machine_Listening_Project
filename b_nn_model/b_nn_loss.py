import torch
from torch import nn


class CosineSimilarity01(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineSimilarity01, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        cos_sim = torch.nn.functional.cosine_similarity(x, y, dim=1)
        return cos_sim


def cosine_similarity_matrix(x, y):
    """
    x: (B_x, D)
    y: (B_y, D)
    output: (B_x, B_y)
    """
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
    sim_matrix = torch.matmul(x_norm, y_norm.T)  # (B_x, B_y)
    return sim_matrix


def matrix_bce_loss_balanced_v2(sim_matrix, eps=1e-8):
    B = sim_matrix.size(0)
    sim_matrix = sim_matrix.clamp(min=eps, max=1 - eps)

    labels = torch.eye(B, device=sim_matrix.device).bool()

    pos_loss = -torch.log(sim_matrix[labels])
    neg_loss = -torch.log(1 - sim_matrix[~labels])

    return 0.5 * pos_loss.mean() + 0.5 * neg_loss.mean()


class LearnableAdaCosLoss(nn.Module):
    def __init__(self, initial_s=1.0, eps=1e-8):
        super(LearnableAdaCosLoss, self).__init__()
        self.s = nn.Parameter(torch.tensor(initial_s))  # 可学习 scale
        self.eps = eps

    def forward(self, att, x):
        # Compute cosine similarity
        cosine = cosine_similarity_matrix(att, x)  # [B, B]

        # Create ground truth (diagonal elements are positive pairs)
        labels = torch.arange(att.shape[0], device=att.device)

        # Scale similarity and compute cross-entropy
        logits = self.s * cosine
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')

        return loss