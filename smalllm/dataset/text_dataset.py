import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from smalllm.config import Config
from smalllm.tokenizer import Tokenizer


class TextDataset(Dataset):
    def __init__(
        self,
        text_data: str,
        tokenizer: Tokenizer,
        context_size: int,  # context window
        stride: int,  # sliding window steps
    ):
        self.token_ids = tokenizer.encode(text_data)
        self.context_size = context_size

        # Separate input and target tokens
        self.inputs = []
        self.targets = []
        for i in tqdm(
            range(0, len(self.token_ids) - context_size + 1, stride), "loader"
        ):
            self.inputs.append(
                torch.tensor(self.token_ids[i : i + context_size], dtype=torch.long)
            )
            self.targets.append(
                torch.tensor(
                    self.token_ids[i + 1 : i + context_size + 1], dtype=torch.long
                )
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def create_data_loaders(
    data: str,
    tokenizer: Tokenizer,
    config: Config,
    val_split: float = 0.1,
):
    """
    Splits the data into training and validation sets and returns DataLoaders for each.
    """
    token_ids = tokenizer.encode(data)
    split_idx = int(len(token_ids) * (1 - val_split))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_dataset = TextDataset(
        tokenizer.decode(train_ids), tokenizer, config.context_size, config.stride
    )
    val_dataset = TextDataset(
        tokenizer.decode(val_ids), tokenizer, config.context_size, config.stride
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, val_loader
