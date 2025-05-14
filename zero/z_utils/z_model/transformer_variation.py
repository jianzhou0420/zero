import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLN(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(embedding_dim, 2 * embedding_dim, bias=True)
        )
        nn.init.constant_(self.modulation[-1].weight, 0)  # identity at the begining
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x, t):
        """
        Args:
            x: A tensor of shape (N, B, C)
            t: A tensor of shape (B, C)
        """
        scale, shift = self.modulation(t).chunk(2, dim=-1)  # (B, C), (B, C)
        x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        return x


class FeedforwardLayer(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, dropout=0.0,
                 use_adaln=False):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        self._reset_parameters()
        if use_adaln:
            self.adaln = AdaLN(embedding_dim)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, diff_ts=None):
        if diff_ts is not None:
            x = self.adaln(x, diff_ts)
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = x + self.dropout(output)
        output = self.norm(output)
        return output
