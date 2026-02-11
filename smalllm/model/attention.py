import torch
import torch.nn as nn
import torch.nn.functional as F

from smalllm.config import Config


class CausalSelfAttention(nn.Module):
    """
    Efficient multi-head causal self-attention.
    """

    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        # Single projections for all heads combined
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.context_size, config.context_size)).to(
                torch.bool
            ),
        )

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        # Project Q, K, V for all heads at once
        q = self.q_proj(x)  # (B, T, n_embd)
        k = self.k_proj(x)  # (B, T, n_embd)
        v = self.v_proj(x)  # (B, T, n_embd)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Concatenate with cached keys/values if provided
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)  # (B, n_head, S, head_dim)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v)
        S = k.shape[2]  # full sequence length (cached + new)

        # Compute attention scores
        # mask shape: (1, 1, T, S) — broadcast over batch and heads
        mask = self.tril[:S, :S][-T:, :].unsqueeze(0).unsqueeze(0)
        wei = q @ k.transpose(-2, -1) * self.head_dim**-0.5  # (B, n_head, T, S)
        wei = wei.masked_fill(mask == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        # Weighted aggregation
        out = wei @ v  # (B, n_head, T, head_dim)

        # Reshape back: (B, T, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))

        return out, new_cache
