from dataclasses import dataclass

import torch


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # dataset
    vocab_size: int = 50257
    context_size: int = 512
    stride: int = 1
    train_data_path: str = r"E:\dev\Machine Learning\smallLM\data\data.txt"
    # model
    n_embd: int = 128
    n_head: int = 8
    n_layer: int = 8
    dropout: float = 0.1
    batch_size: int = 32
    max_iters: int = 5000
    warmup_steps: int = 500
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 1e-3
    min_lr: float = 5e-3
    best_model_path: str = "checkpoints/best_model.pt"
