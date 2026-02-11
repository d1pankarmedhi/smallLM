import torch.nn as nn

from smalllm.config import Config
from smalllm.model.attention import CausalSelfAttention
from smalllm.model.mlp import MLP


class TransformerBlock(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config: Config):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.self_attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.self_attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache
