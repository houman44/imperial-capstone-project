import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        N = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        queries = self.query_linear(query).view(N, query_len, self.heads, self.head_dim)
        keys = self.key_linear(key).view(N, key_len, self.heads, self.head_dim)
        values = self.value_linear(value).view(N, value_len, self.heads, self.head_dim)
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, values)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(N, query_len, self.embed_size)
        out = self.fc_out(out)
        return out
