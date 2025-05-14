import torch
import torch.nn as nn
import torch.nn.functional as F


class FFWandAddNorm(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x,):
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = x + self.dropout(output)
        output = self.norm(output)
        return output


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)

    def forward(self, x, y, mask=None):
        attn_output, _ = self.multihead_attn(x, y, y, key_padding_mask=mask)
        return attn_output


class SelfAttnFFW(nn.Module):
    def __init__(self, d_model, n_heads, d_ffw, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, n_heads, dropout)
        self.ffw = FFWandAddNorm(d_model, d_ffw, dropout)

    def forward(self, x):
        attn_output = self.self_attn(x)
        ffw_output = self.ffw(attn_output)
        return ffw_output


class CrossAttnFFW(nn.Module):
    def __init__(self, d_model, n_heads, d_ffw, dropout=0.1):
        super().__init__()
        self.cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)
        self.ffw = FFWandAddNorm(d_model, d_ffw, dropout)

    def forward(self, x, y, mask=None):
        attn_output = self.cross_attn(x, y, mask)
        ffw_output = self.ffw(attn_output)
        return ffw_output


class PositionalEncoding(nn.Module):
    '''
    1D PE: Sequence PE
    Assume we have embedding a sentence to shape [len(sentence), d_model] where d_model is the dimension of embedding. one world one embedding vector.
    embedding vector has d_model dimension.
    Positional encoding is to use a formula to add some values to each dimension of embedding vector. The specific value to add on each element of embedding vector is determined by the position of the element in the sentence.
    depends on the position of the word and the dimension of the embedding.

    '''

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # we are goona create a PE table. It is a tensor with shape [max_len, d_model]
        PE = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len)
        row = torch.pow(10000, 2 * (torch.arange(0, d_model) // 2) / d_model)
        PE[:, 0::2] = torch.sin(position.unsqueeze(1) / row[0::2])
        PE[:, 1::2] = torch.cos(position.unsqueeze(1) / row[1::2])
        self.register_buffer('PE', PE)

    def forward(self, x):
        # assume x has shape of [batch_size, len(sentence), d_model]
        return x + self.PE[:x.size(1), :]
