import torch
import torch.nn as nn
import torch.nn.functional as F

from smalllm.config import Config
from smalllm.model.transformer import TransformerBlock


class LanguageModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.context_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None, start_pos=0):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        positions = torch.arange(start_pos, start_pos + T, device=idx.device)
        pos_emb = self.position_embedding_table(positions)  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_new_cache = block(x, kv_cache=layer_cache)
            new_kv_cache.append(layer_new_cache)

        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, new_kv_cache

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, use_kv_cache=False):
        self.eval()
        # idx is (B, T) array of indices in the current context

        if not use_kv_cache:
            # full recomputation each step
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.context_size :]
                logits, loss, _ = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        else:
            # KV-cached generation: prefill + incremental decoding
            kv_cache = None
            # Prefill: process all existing tokens
            idx_cond = idx[:, -self.config.context_size :]
            prefill_len = idx_cond.shape[1]
            logits, _, kv_cache = self(idx_cond, kv_cache=None, start_pos=0)
            logits = logits[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            cache_pos = prefill_len  # number of tokens currently in the cache

            # Incremental decoding: one token at a time
            for _ in range(max_new_tokens - 1):
                if cache_pos >= self.config.context_size:
                    # Re-prefill: slide the window and rebuild the cache
                    idx_cond = idx[:, -self.config.context_size :]
                    logits, _, kv_cache = self(idx_cond, kv_cache=None, start_pos=0)
                    cache_pos = idx_cond.shape[1]  # reset to prefill length
                else:
                    logits, _, kv_cache = self(
                        idx_next, kv_cache=kv_cache, start_pos=cache_pos
                    )
                    cache_pos += 1  # new token added to cache
                logits = logits[:, -1, :]
                probs = F.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)

        return idx
