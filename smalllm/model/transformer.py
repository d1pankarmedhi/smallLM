import torch.nn as nn

from smalllm.config import Config
from smalllm.model.mha import MultiHeadAttention
from smalllm.model.mlp import MLP


class TransformerBlock(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config: Config):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
