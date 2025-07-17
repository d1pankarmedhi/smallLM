import torch
import torch.nn as nn

from smalllm.config import Config
from smalllm.model.attention import CausalAttentionHead


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, config: Config):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttentionHead(config) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
